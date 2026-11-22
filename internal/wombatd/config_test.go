package wombatd

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"wombat/internal/yaml"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "wombatd.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// A file names only what it changes, and everything else keeps its default.
func TestAConfigNamesOnlyWhatItChanges(t *testing.T) {
	cfg, err := loadConfig(writeConfig(t, `
model:
  provider: anthropic
  name: claude-sonnet-4-5
  api_key: sk-test
  headers:
    X-Client: wombat
turn:
  wall: 10m
done:
  check: go test ./...
review:
  enabled: true
  sweeps: 6
`))
	if err != nil {
		t.Fatal(err)
	}
	want := defaults()
	want.Model.Provider, want.Model.Name, want.Model.APIKey = "anthropic", "claude-sonnet-4-5", "sk-test"
	want.Model.Headers = map[string]string{"X-Client": "wombat"}
	want.Turn.Wall = 10 * time.Minute
	want.Done.Check = "go test ./..."
	want.Review.Enabled, want.Review.Sweeps = true, 6
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("got  %+v\nwant %+v", cfg, want)
	}
}

// Every refusal names the file, and the key or the line it is about, in words
// the person editing the file can act on.
func TestAConfigThatCannotMeanWhatItSaysIsRefused(t *testing.T) {
	for _, tc := range []struct{ name, body, want string }{
		{"a misspelt key", "review:\n  sweep: 6\n", `line 2: unknown key "sweep" in review`},
		{"a word that is not a choice", "server:\n  log: loud\n", `server.log: "loud" is not one of debug, error, info, warn`},
		{"an unknown provider", "model:\n  provider: gemini\n", "model.provider"},
		{"an unknown permission", "tools:\n  permission: read-only\n", "tools.permission"},
		{"an unknown network", "tools:\n  confine_network: some\n", "tools.confine_network"},
		{"a relative path to hide", "tools:\n  confine: true\n  deny_read: [secrets]\n", `tools.deny_read: "secrets" is not an absolute path`},
		{"paths to hide, unconfined", "tools:\n  deny_read: [/etc/wombat]\n", "tools.deny_read: is applied by tools.confine, which is off"},
		{"an empty tool to disable", "tools:\n  disable: [write_file, \"\"]\n", "tools.disable: holds an empty name"},
		{"an unknown recovery", "tools:\n  recovery: retry\n", `tools.recovery: "retry" is not one of ask, report`},
		{"a negative retention", "store:\n  ledger_retention: -1h\n", "store.ledger_retention: is negative"},
		{"a negative slice", "daemon:\n  slice: -1s\n", "daemon.slice: is negative"},
		{"a wake-up that is not a URL", "serverless:\n  wake:\n    url: queue\n", `serverless.wake.url: "queue" is not an http or https URL`},
		{"wake-up headers without a URL", "serverless:\n  wake:\n    delay_header: Upstash-Not-Before\n", "serverless.wake: headers are sent to serverless.wake.url"},
		{"an unknown mode", "mode: plan-only\n", `mode: "plan-only" is not one of build, plan, review`},
		{"compaction without a window", "turn:\n  compact_at: 0.8\n", "turn.context_window, which is not set"},
		{"compaction past the window", "turn:\n  context_window: 100000\n  compact_at: 1.5\n", "not a fraction"},
		{"a report with no trace", "record:\n  trace_html: t.html\n", "the record.trace file, which is not set"},
		{"no iterations", "turn:\n  max_iters: 0\n", "turn.max_iters: 0 is less than 1"},
		{"a duration without a unit", "turn:\n  wall: 90\n", "turn.wall"},
		{"a stream that is not a word", "model:\n  stream: false\n", "model.stream"},
		{"a bad tape mode", "record:\n  tape_mode: rewind\n", "record.tape_mode"},
		{"a number in quotes", "turn:\n  max_iters: \"20\"\n", "quoted"},
		{"an unknown store", "store:\n  driver: sqlite\n", `store.driver: "sqlite" is not one of memory, postgres`},
		{"postgres without a dsn", "store:\n  driver: postgres\n", "store.dsn: is empty"},
		{"a dsn for memory", "store:\n  dsn: postgres://x\n", "store.driver is memory"},
		{"no runs", "daemon:\n  max_runs: 0\n", "daemon.max_runs: 0 is less than 1"},
		{"no poll", "daemon:\n  poll: 0s\n", "daemon.poll"},
		{"a webhook that is not a URL", "outbox:\n  webhook: localhost:9000\n", "outbox.webhook"},
		{"headers without a webhook", "outbox:\n  headers:\n    Authorization: x\n", "outbox.webhook, which is not set"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeConfig(t, tc.body)
			_, err := loadConfig(path)
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), path+": ") || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v\nwant it to name %s and say %q", err, path, tc.want)
			}
		})
	}
}

func TestAMissingConfigSaysWhereToStart(t *testing.T) {
	_, err := loadConfig(filepath.Join(t.TempDir(), "wombatd.yaml"))
	if err == nil || !strings.Contains(err.Error(), "wombatd.example.yaml") {
		t.Errorf("err = %v, want it to point at the example", err)
	}
}

// TestTheExampleConfigShowsEveryKeyAndItsDefault: wombatd.example.yaml is
// where the keys are documented, so a key missing from it is a key nobody
// knows about, and a default it misstates is worse than none. Outside the
// model section, whose values are placeholders to fill in, every value it
// shows is the default.
func TestTheExampleConfigShowsEveryKeyAndItsDefault(t *testing.T) {
	const example = "../../wombatd.example.yaml"
	cfg, err := loadConfig(example)
	if err != nil {
		t.Fatal(err)
	}
	want := defaults()
	want.Model = cfg.Model
	if len(cfg.Server.CORS) == 0 {
		cfg.Server.CORS = nil // [] in the file, nil by default: the same empty list
	}
	if len(cfg.Outbox.Headers) == 0 {
		cfg.Outbox.Headers = nil // {} in the file, nil by default
	}
	if len(cfg.Tools.DenyRead) == 0 {
		cfg.Tools.DenyRead = nil // [] in the file, nil by default
	}
	if len(cfg.Tools.Disable) == 0 {
		cfg.Tools.Disable = nil // [] in the file, nil by default
	}
	if len(cfg.Serverless.Wake.Headers) == 0 {
		cfg.Serverless.Wake.Headers = nil // {} in the file, nil by default
	}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("the example's values are not the defaults:\nexample  %+v\ndefaults %+v", cfg, want)
	}

	data, err := os.ReadFile(example)
	if err != nil {
		t.Fatal(err)
	}
	var shown map[string]any
	if err := yaml.Unmarshal(data, &shown); err != nil {
		t.Fatal(err)
	}
	for _, key := range keysOf(reflect.TypeFor[config](), "") {
		if !has(shown, key) {
			t.Errorf("the example does not show %s", key)
		}
	}
}

// keysOf lists the dotted path of every key a configuration file can set.
func keysOf(t reflect.Type, prefix string) []string {
	var keys []string
	for i := range t.NumField() {
		f := t.Field(i)
		key := prefix + f.Tag.Get("yaml")
		if f.Type.Kind() == reflect.Struct {
			keys = append(keys, keysOf(f.Type, key+".")...)
			continue
		}
		keys = append(keys, key)
	}
	return keys
}

func has(m map[string]any, dotted string) bool {
	first, rest, nested := strings.Cut(dotted, ".")
	v, ok := m[first]
	if !ok || !nested {
		return ok
	}
	sub, ok := v.(map[string]any)
	return ok && has(sub, rest)
}

// The file wombatd was configured from is hidden from confined commands,
// however it was named: it holds what only wombatd may read, and the commands
// run as wombatd's user, whom its permissions let in.
func TestTheConfigFileIsHiddenFromCommands(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cfg := defaults()
	cfg.Tools.DenyRead = []string{"/etc/wombat"}
	for _, tc := range []struct{ named, want string }{
		{"wombatd.yaml", filepath.Join(cwd, "wombatd.yaml")},
		{filepath.Join(dir, "conf", "w.yaml"), filepath.Join(dir, "conf", "w.yaml")},
	} {
		got, err := cfg.hidingFile(tc.named)
		if err != nil {
			t.Fatal(err)
		}
		if want := []string{"/etc/wombat", tc.want}; !slices.Equal(got.Tools.DenyRead, want) {
			t.Errorf("hidingFile(%q).Tools.DenyRead = %q, want %q", tc.named, got.Tools.DenyRead, want)
		}
	}
	if len(cfg.Tools.DenyRead) != 1 {
		t.Errorf("hidingFile changed the config it was called on: %q", cfg.Tools.DenyRead)
	}
}

// A secret in a file anyone else can read is said out loud; a file without
// one, or one only its owner can read, is not.
func TestASecretOthersCanReadIsWarnedAbout(t *testing.T) {
	path := writeConfig(t, "")
	for _, tc := range []struct {
		mode   os.FileMode
		secret string
		warns  bool
	}{
		{0o644, "sk-test", true},
		{0o600, "sk-test", false},
		{0o644, "", false},
	} {
		if err := os.Chmod(path, tc.mode); err != nil {
			t.Fatal(err)
		}
		cfg := defaults()
		cfg.Model.APIKey = tc.secret
		var log bytes.Buffer
		warnIfReadable(path, cfg, slog.New(slog.NewTextHandler(&log, nil)))
		if warned := strings.Contains(log.String(), "chmod 600"); warned != tc.warns {
			t.Errorf("mode %v, secret %q: warned = %v, want %v", tc.mode, tc.secret, warned, tc.warns)
		}
	}
}
