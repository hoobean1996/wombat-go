package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMediaSurvivesTheMessageJSONRoundTrip(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\nnot really")
	in := []Message{
		{Role: RoleUser, Content: []ContentBlock{
			Text{Text: "what is in this?"},
			Media{MediaType: "image/png", Data: png, Name: "shot.png"},
		}},
		{Role: RoleUser, Content: []ContentBlock{
			ToolResult{ToolUseID: "u1", Content: "Attached shot.png", Media: []Media{{MediaType: "application/pdf", Data: []byte("%PDF-1.7")}}},
		}},
	}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out []Message
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("round trip changed the messages:\n in  %#v\n out %#v\n json %s", in, out, raw)
	}
}

// A text-only message must encode exactly as it did before media existed:
// persisted transcripts and tape keys depend on those bytes.
func TestTextOnlyMessageJSONIsUnchangedByMedia(t *testing.T) {
	raw, err := json.Marshal(Message{Role: RoleUser, Content: []ContentBlock{
		Text{Text: "hi"}, ToolResult{ToolUseID: "u1", Content: "ok"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"role":"user","content":[{"type":"text","text":"hi"},{"type":"tool_result","tool_use_id":"u1","content":"ok"}]}`
	if string(raw) != want {
		t.Errorf("got  %s\nwant %s", raw, want)
	}
}

func TestMediaIsImage(t *testing.T) {
	if !(Media{MediaType: "image/webp"}).IsImage() || (Media{MediaType: "application/pdf"}).IsImage() {
		t.Error("IsImage misclassifies")
	}
}
