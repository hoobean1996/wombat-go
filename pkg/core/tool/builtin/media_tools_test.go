package builtin

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

func viewMedia(t *testing.T, path string) tool.Result {
	t.Helper()
	d := ViewMedia(OSFS(""))
	in, _ := json.Marshal(map[string]string{"path": path})
	return tool.NewDispatcher(tool.Direct).Dispatch(context.Background(), tool.NewSet(d),
		[]llm.ToolUse{{ID: "u1", Name: d.Name, Input: in}})[0]
}

func TestViewMediaAttachesAnImage(t *testing.T) {
	png := append([]byte("\x89PNG\r\n\x1a\n"), bytes.Repeat([]byte{0}, 32)...)
	// Named .jpg on purpose: the bytes decide the type, not the name.
	path := filepath.Join(t.TempDir(), "shot.jpg")
	if err := os.WriteFile(path, png, 0o644); err != nil {
		t.Fatal(err)
	}
	res := viewMedia(t, path)
	if res.Err != nil {
		t.Fatal(res.Err)
	}
	if len(res.Media) != 1 || res.Media[0].MediaType != "image/png" || !bytes.Equal(res.Media[0].Data, png) {
		t.Fatalf("Media = %+v, want the file as image/png", res.Media)
	}
	if !strings.Contains(res.Output, "image/png") {
		t.Errorf("Output = %q, want it to say what was attached", res.Output)
	}
}

func TestViewMediaAttachesAPDF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spec.pdf")
	if err := os.WriteFile(path, []byte("%PDF-1.7\n%..."), 0o644); err != nil {
		t.Fatal(err)
	}
	res := viewMedia(t, path)
	if res.Err != nil || len(res.Media) != 1 || res.Media[0].MediaType != "application/pdf" || res.Media[0].Name != "spec.pdf" {
		t.Fatalf("got %+v, %v; want the PDF attached under its name", res.Media, res.Err)
	}
}

func TestViewMediaRefusesWhatIsNotMedia(t *testing.T) {
	dir := t.TempDir()
	text := filepath.Join(dir, "notes.png")
	if err := os.WriteFile(text, []byte("just text"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{text, dir, filepath.Join(dir, "missing.png"), "relative.png"} {
		if res := viewMedia(t, p); res.Err == nil || len(res.Media) != 0 {
			t.Errorf("%s: got %v with %d attachments, want a refusal", p, res.Err, len(res.Media))
		}
	}
}

func TestViewMediaRefusesAnOversizedImage(t *testing.T) {
	big := append([]byte("\x89PNG\r\n\x1a\n"), make([]byte, maxImageBytes)...)
	path := filepath.Join(t.TempDir(), "big.png")
	if err := os.WriteFile(path, big, 0o644); err != nil {
		t.Fatal(err)
	}
	if res := viewMedia(t, path); res.Err == nil || !strings.Contains(res.Err.Error(), "limit") {
		t.Errorf("err = %v, want the size limit", res.Err)
	}
}

// ViewMedia is not part of Default: a text-only model given an image fails the
// whole turn.
func TestViewMediaIsNotADefaultTool(t *testing.T) {
	for _, d := range Default(Deps{}) {
		if d.Name == "view_media" {
			t.Fatal("view_media is in Default")
		}
	}
}
