package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// Def turns one of the server's tools into a tool.Def the agent can dispatch.
// Its name is <server>__<tool>, made safe for the providers; see the package
// doc for its capabilities.
func (c *Client) Def(t Tool) tool.Def {
	caps, idempotent := c.caps(t.Annotations)
	desc := strings.TrimSpace(t.Description)
	if desc == "" {
		desc = t.Title
	}
	if desc == "" && t.Annotations != nil {
		desc = t.Annotations.Title
	}
	schema := t.InputSchema
	if len(schema) == 0 || string(schema) == "null" {
		schema = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	remote := t.Name
	return tool.Def{
		Name:        toolName(c.server.Name, t.Name),
		Description: desc,
		InputSchema: schema,
		Caps:        caps,
		Idempotent:  idempotent,
		Timeout:     c.server.Timeout,
		Category:    "mcp:" + c.server.Name,
		Fn: func(ctx context.Context, in json.RawMessage) (string, error) {
			return c.CallTool(ctx, remote, in)
		},
	}
}

// caps reads a tool's annotations the way the specification defines their
// defaults: not read-only, and reaching the open world, unless it says so.
func (c *Client) caps(a *Annotations) (caps tool.Cap, idempotent bool) {
	if c.server.IgnoreHints || a == nil {
		return tool.CapMutating | tool.CapNetwork, false
	}
	readOnly := a.ReadOnlyHint != nil && *a.ReadOnlyHint
	if readOnly {
		caps = tool.CapReadOnly
	} else {
		caps = tool.CapMutating
	}
	if a.OpenWorldHint == nil || *a.OpenWorldHint {
		caps |= tool.CapNetwork
	}
	return caps, readOnly || (a.IdempotentHint != nil && *a.IdempotentHint)
}

// CallTool calls the server's tool name and returns its result the way a
// tool.Fn does: the text, with any images and documents attached to the call
// ctx belongs to, and an error when the call failed or the tool reported one.
func (c *Client) CallTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
	if len(args) == 0 || string(args) == "null" {
		args = json.RawMessage(`{}`)
	}
	var res callResult
	params := map[string]any{"name": name, "arguments": args}
	if id := sessionFrom(ctx); c.server.Session && id != "" {
		params["_meta"] = map[string]any{MetaSession: id}
	}
	err := c.call(ctx, "tools/call", params, &res)
	if err != nil {
		var re *rpcError
		if errors.As(err, &re) && re.Code == errInvalidParams {
			// The server refused the arguments, or does not know the tool:
			// the call's fault, not a sign the server is unwell.
			return "", fmt.Errorf("%w: %s", tool.ErrInvalidInput, re.Message)
		}
		return "", fmt.Errorf("mcp server %s: %w", c.server.Name, err)
	}
	text, media := render(res)
	if res.IsError {
		if text == "" {
			text = "the tool reported an error and said nothing about it"
		}
		// The server ran the call and answered: what failed is the request,
		// as a command that exits non-zero is.
		return "", tool.CallerError(errors.New(text))
	}
	tool.Attach(ctx, media...)
	return text, nil
}

type callResult struct {
	Content           []content       `json:"content"`
	StructuredContent json.RawMessage `json:"structuredContent,omitempty"`
	IsError           bool            `json:"isError,omitempty"`
}

type content struct {
	Type        string           `json:"type"`
	Text        string           `json:"text,omitempty"`
	Data        string           `json:"data,omitempty"`
	MimeType    string           `json:"mimeType,omitempty"`
	URI         string           `json:"uri,omitempty"`
	Name        string           `json:"name,omitempty"`
	Description string           `json:"description,omitempty"`
	Resource    *embeddedContent `json:"resource,omitempty"`
}

type embeddedContent struct {
	URI      string  `json:"uri"`
	MimeType string  `json:"mimeType,omitempty"`
	Text     *string `json:"text,omitempty"`
	Blob     string  `json:"blob,omitempty"`
}

// What the providers accept as one image or one document.
const (
	maxImageBytes = 5 << 20
	maxPDFBytes   = 10 << 20
)

// render turns a tool result into the text the model reads and the media it
// is shown. Anything that cannot be shown is still mentioned, so the model
// knows the server returned it.
func render(res callResult) (string, []llm.Media) {
	var parts []string
	var media []llm.Media
	add := func(mimeType, b64, name string) {
		m, note := decodeMedia(mimeType, b64, name)
		if m != nil {
			media = append(media, *m)
		}
		parts = append(parts, note)
	}
	for i, ct := range res.Content {
		switch ct.Type {
		case "text":
			parts = append(parts, ct.Text)
		case "image", "audio":
			add(ct.MimeType, ct.Data, fmt.Sprintf("%s %d", ct.Type, i+1))
		case "resource":
			r := ct.Resource
			switch {
			case r == nil:
			case r.Text != nil:
				parts = append(parts, fmt.Sprintf("[resource %s]\n%s", r.URI, *r.Text))
			default:
				add(r.MimeType, r.Blob, r.URI)
			}
		case "resource_link":
			line := "[resource link: " + ct.URI
			if ct.Name != "" {
				line += " (" + ct.Name + ")"
			}
			if ct.Description != "" {
				line += " — " + ct.Description
			}
			parts = append(parts, line+"]")
		default:
			parts = append(parts, fmt.Sprintf("[%s content omitted: this client does not understand it]", ct.Type))
		}
	}
	// A server is told to send its structured result as text as well; one
	// that sends nothing else still has it read.
	if len(res.Content) == 0 && len(res.StructuredContent) > 0 {
		parts = append(parts, string(res.StructuredContent))
	}
	return strings.Join(parts, "\n"), media
}

// decodeMedia returns what can be shown to the model, and the line of text
// that says what it was.
func decodeMedia(mimeType, b64, name string) (*llm.Media, string) {
	if !slices.Contains(llm.MediaTypes, mimeType) {
		return nil, fmt.Sprintf("[%s (%s) omitted: not a kind of content the model can be shown]", name, mimeType)
	}
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Sprintf("[%s omitted: its data is not valid base64]", name)
	}
	limit := maxImageBytes
	if mimeType == "application/pdf" {
		limit = maxPDFBytes
	}
	if len(data) > limit {
		return nil, fmt.Sprintf("[%s (%s, %d bytes) omitted: larger than the %d bytes a model is shown]", name, mimeType, len(data), limit)
	}
	return &llm.Media{MediaType: mimeType, Data: data, Name: name},
		fmt.Sprintf("[attached %s: %s, %d bytes]", name, mimeType, len(data))
}

// ===== names =====

// maxName is the longest tool name both providers accept.
const maxName = 64

// toolName is the name a server's tool is offered to the model under: the
// server's name and the tool's joined by two underscores, every byte the
// providers refuse replaced by one, and the whole cut to fit with a hash of
// the original standing in for what was cut.
func toolName(server, name string) string {
	n := sanitize(server) + "__" + sanitize(name)
	if len(n) <= maxName {
		return n
	}
	return hashed(n, server, name)
}

func hashed(n, server, name string) string {
	sum := sha256.Sum256([]byte(server + "\x00" + name))
	suffix := "_" + hex.EncodeToString(sum[:4])
	return n[:min(len(n), maxName-len(suffix))] + suffix
}

// uniqueName is toolName, made distinct from every name in taken — two tools
// whose names differ only in bytes sanitize replaced would otherwise collide,
// and a tool set refuses duplicates — and added to it.
func uniqueName(taken map[string]bool, server, name string) string {
	base := toolName(server, name)
	n := base
	for i := 1; taken[n]; i++ {
		n = hashed(base, server, fmt.Sprintf("%s\x00%d", name, i))
	}
	taken[n] = true
	return n
}

func sanitize(s string) string {
	b := []byte(s)
	for i, c := range b {
		if !('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c == '_' || c == '-') {
			b[i] = '_'
		}
	}
	return string(b)
}
