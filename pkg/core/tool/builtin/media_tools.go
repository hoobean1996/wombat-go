package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// Media size caps. An image past 5 MiB is refused by the providers anyway, and
// a PDF is capped well below the request limit once base64 has inflated it by a
// third.
const (
	maxImageBytes = 5 << 20
	maxPDFBytes   = 10 << 20
)

type viewMediaIn struct {
	Path string `json:"path"`
}

// ViewMedia returns view_media, which shows the model an image or a PDF from
// the filesystem: a screenshot a test wrote, a diagram in the docs, a spec
// that only exists as a PDF.
//
// It is NOT in [Default], and that is deliberate. A model that cannot read
// images answers a request carrying one with a 400, and the turn — and so the
// run — fails over a tool call the model was invited to make. Add it to an
// agent whose model reads media.
//
// The type is decided from the bytes, not the extension, and a file whose
// bytes are not an image or a PDF is refused rather than sent under a type it
// does not have.
func ViewMedia(fsys FS) tool.Def {
	mustNotBeNil(fsys != nil, "ViewMedia requires a non-nil FS")

	return tool.Typed(tool.Def{
		Name: "view_media",
		Description: "Look at an image (PNG, JPEG, GIF, WebP) or a PDF document. The file is attached " +
			"to the result so you can see it. Images up to 5 MiB, PDFs up to 10 MiB. " +
			"PATH MUST BE ABSOLUTE (starts with /).",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path of the image or PDF"
    }
  },
  "required": ["path"]
}`),
		Caps:       tool.CapReadOnly,
		Needs:      tool.NeedFSRead,
		Idempotent: true,
		Timeout:    10 * time.Second,
		Category:   "file_io",
		Retryable:  retryFS,
	}, func(ctx context.Context, in viewMediaIn) (string, error) {
		if err := requireAbs("path", in.Path); err != nil {
			return "", err
		}
		kind, err := fsys.Stat(ctx, in.Path)
		if err != nil {
			return "", err
		}
		switch kind {
		case KindMissing:
			return "", tool.CallerError(fmt.Errorf("no such file: %s", in.Path))
		case KindDir:
			return "", tool.CallerError(fmt.Errorf("%s is a directory; view_media shows one file", in.Path))
		}
		data, err := fsys.ReadFile(ctx, in.Path)
		if err != nil {
			return "", err
		}
		mediaType, err := sniffMedia(in.Path, data)
		if err != nil {
			return "", tool.CallerError(err)
		}
		limit := maxImageBytes
		if mediaType == "application/pdf" {
			limit = maxPDFBytes
		}
		if len(data) > limit {
			return "", tool.CallerError(fmt.Errorf("%s is %d bytes, over the %d-byte limit for %s",
				in.Path, len(data), limit, mediaType))
		}
		tool.Attach(ctx, llm.Media{MediaType: mediaType, Data: data, Name: filepath.Base(in.Path)})
		return fmt.Sprintf("Attached %s (%s, %d bytes).", in.Path, mediaType, len(data)), nil
	})
}

// sniffMedia decides what data is. Content sniffing is the authority — a .png
// that is really a JPEG is sent as a JPEG — and the result must be one of the
// types the providers accept.
func sniffMedia(path string, data []byte) (string, error) {
	t := http.DetectContentType(data)
	if i := strings.IndexByte(t, ';'); i >= 0 {
		t = t[:i]
	}
	if slices.Contains(llm.MediaTypes, t) {
		return t, nil
	}
	return "", fmt.Errorf("%s is %s, not an image (PNG, JPEG, GIF, WebP) or a PDF", path, t)
}
