package mcp

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
	"time"
)

// LoadConfig reads servers from a JSON file in the format desktop MCP clients
// use, sorted by name:
//
//	{"mcpServers": {
//	  "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
//	             "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}},
//	  "docs":   {"type": "http", "url": "https://example.com/mcp",
//	             "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}
//	}}
//
// "${NAME}" in a command, an argument, an env or header value, a cwd or a url
// is replaced by the harness's own environment variable NAME, and
// "${NAME:-fallback}" by fallback when NAME is unset; a variable that is unset
// with no fallback is an error rather than an empty string, since a server
// started without its token fails later and less clearly. A server with
// "disabled": true is skipped, "ignoreHints": true sets [Server.IgnoreHints],
// "session": true sets [Server.Session] (the server is told the calling
// session), and "timeout" sets [Server.Timeout] as a Go duration ("90s", "10m").
func LoadConfig(path string) ([]Server, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("mcp: %w", err)
	}
	servers, err := ParseConfig(raw)
	if err != nil {
		return nil, fmt.Errorf("%w (in %s)", err, path)
	}
	return servers, nil
}

// ParseConfig is [LoadConfig] on the file's contents.
func ParseConfig(raw []byte) ([]Server, error) {
	var file struct {
		Servers map[string]serverConfig `json:"mcpServers"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("mcp: %w", err)
	}
	names := make([]string, 0, len(file.Servers))
	for name := range file.Servers {
		names = append(names, name)
	}
	slices.Sort(names)

	var out []Server
	for _, name := range names {
		sc := file.Servers[name]
		if sc.Disabled {
			continue
		}
		s, err := sc.server(name)
		if err != nil {
			return nil, fmt.Errorf("mcp: server %q: %w", name, err)
		}
		out = append(out, s)
	}
	return out, nil
}

type serverConfig struct {
	Type        string            `json:"type"`
	Command     string            `json:"command"`
	Args        []string          `json:"args"`
	Env         map[string]string `json:"env"`
	Cwd         string            `json:"cwd"`
	URL         string            `json:"url"`
	Headers     map[string]string `json:"headers"`
	Disabled    bool              `json:"disabled"`
	IgnoreHints bool              `json:"ignoreHints"`
	Session     bool              `json:"session"`
	Timeout     string            `json:"timeout"`
}

func (sc serverConfig) server(name string) (Server, error) {
	if name == "" {
		return Server{}, fmt.Errorf("a server needs a name")
	}
	switch strings.ToLower(sc.Type) {
	case "":
	case "stdio":
		if sc.Command == "" {
			return Server{}, fmt.Errorf(`type "stdio" needs a command`)
		}
	case "http", "streamable-http", "streamablehttp":
		if sc.URL == "" {
			return Server{}, fmt.Errorf("type %q needs a url", sc.Type)
		}
	case "sse":
		return Server{}, fmt.Errorf(`type "sse" is the HTTP+SSE transport of the 2024-11-05 revision, ` +
			`which this client does not speak; point it at the server's Streamable HTTP endpoint with type "http"`)
	default:
		return Server{}, fmt.Errorf("unknown type %q (want stdio or http)", sc.Type)
	}
	if (sc.Command == "") == (sc.URL == "") {
		return Server{}, fmt.Errorf("needs exactly one of a command and a url")
	}

	var timeout time.Duration
	if sc.Timeout != "" {
		d, err := time.ParseDuration(sc.Timeout)
		if err != nil || d < 0 {
			return Server{}, fmt.Errorf("timeout %q is not a duration like \"90s\" or \"10m\"", sc.Timeout)
		}
		timeout = d
	}

	var e expander
	s := Server{
		Timeout:     timeout,
		Name:        name,
		Command:     e.expand(sc.Command),
		Dir:         e.expand(sc.Cwd),
		URL:         e.expand(sc.URL),
		IgnoreHints: sc.IgnoreHints,
		Session:     sc.Session,
	}
	for _, a := range sc.Args {
		s.Args = append(s.Args, e.expand(a))
	}
	s.Env = e.expandMap(sc.Env)
	s.Headers = e.expandMap(sc.Headers)
	return s, e.err
}

var reference = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}`)

// expander replaces ${NAME} references, remembering the first one it could
// not.
type expander struct{ err error }

func (e *expander) expand(s string) string {
	return reference.ReplaceAllStringFunc(s, func(ref string) string {
		m := reference.FindStringSubmatch(ref)
		if v, ok := os.LookupEnv(m[1]); ok {
			return v
		}
		if m[2] != "" {
			return m[3]
		}
		if e.err == nil {
			e.err = fmt.Errorf("${%s} is not set in the environment", m[1])
		}
		return ""
	})
}

func (e *expander) expandMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = e.expand(v)
	}
	return out
}
