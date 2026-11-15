package builtin

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// ===== the environment =====

// TestOSRunnerDoesNotLeakTheHarnessCredentials is the regression test for the
// leak the filter exists for: the provider key in the harness's environment
// was one `echo $OPENAI_API_KEY` — or one test the model wrote — away.
func TestOSRunnerDoesNotLeakTheHarnessCredentials(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	t.Setenv("OPENAI_API_KEY", "sk-leak")
	t.Setenv("GOOGLE_API_KEY", "g-leak")
	t.Setenv("GOFLAGS", "-mod=mod")

	out, err := OSRunner("").Shell(t.Context(), "env")
	if err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{"sk-leak", "g-leak"} {
		if strings.Contains(out, leak) {
			t.Errorf("a credential reached the child:\n%s", out)
		}
	}
	for _, want := range []string{"PATH=", "GOFLAGS=-mod=mod"} {
		if !strings.Contains(out, want) {
			t.Errorf("child environment is missing %s:\n%s", want, out)
		}
	}
}

func TestRunnerEnvOptions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	t.Setenv("WOMBAT_TEST_SECRET", "s")
	t.Setenv("WOMBAT_TEST_PLAIN", "p")
	show := `echo "secret=${WOMBAT_TEST_SECRET-unset} plain=${WOMBAT_TEST_PLAIN-unset} extra=${WOMBAT_TEST_EXTRA-unset}"`

	for _, tc := range []struct {
		name string
		opts []RunnerOption
		want string
	}{
		{"default filters", nil, "secret=unset plain=unset extra=unset"},
		{"allow by name", []RunnerOption{WithEnvAllow("WOMBAT_TEST_PLAIN")}, "secret=unset plain=p extra=unset"},
		{"allow by prefix", []RunnerOption{WithEnvAllow("WOMBAT_TEST_*")}, "secret=s plain=p extra=unset"},
		{"set adds and overrides", []RunnerOption{WithEnv("WOMBAT_TEST_EXTRA=x", "WOMBAT_TEST_SECRET=over")}, "secret=over plain=unset extra=x"},
		{"inherit passes everything", []RunnerOption{WithInheritedEnv()}, "secret=s plain=p extra=unset"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := OSRunner("", tc.opts...).Shell(t.Context(), show)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.TrimSpace(out); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// The filter reads the environment when a command starts, so a variable set
// after the runner was built still arrives. benchmarks.ApplyGoEnv depends on
// it.
func TestOSRunnerReadsTheEnvironmentPerCommand(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	r := OSRunner("")
	t.Setenv("GOPROXY", "off")
	out, err := r.Shell(t.Context(), `echo "$GOPROXY"`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out) != "off" {
		t.Errorf("GOPROXY = %q, want the value set after the runner was built", out)
	}
}

// ===== the plan =====

func testPlan(t *testing.T, sb Sandbox, opts ...RunnerOption) (sandboxPlan, string) {
	t.Helper()
	dir := t.TempDir()
	if sb.CacheDir == "" {
		sb.CacheDir = filepath.Join(t.TempDir(), "cache")
	}
	p, err := planSandbox(dir, sb, opts)
	if err != nil {
		t.Fatal(err)
	}
	return p, realPath(dir)
}

func TestPlanSandbox(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	extra := t.TempDir()
	p, dir := testPlan(t, Sandbox{Writable: []string{extra}, DenyRead: []string{"/secret/place"}})

	for _, want := range []string{dir, realPath(extra), realPath(os.TempDir())} {
		if !slices.Contains(p.writable, want) {
			t.Errorf("writable %q is missing %q", p.writable, want)
		}
	}
	for _, want := range []string{filepath.Join(realPath(home), ".ssh"), filepath.Join(realPath(home), ".aws"), "/secret/place"} {
		if !slices.Contains(p.denied, want) {
			t.Errorf("denied %q is missing %q", p.denied, want)
		}
	}

	// The confined build cache is private, writable and handed to go.
	gocache := ""
	for _, kv := range p.env.sandbox {
		if v, ok := strings.CutPrefix(kv, "GOCACHE="); ok {
			gocache = v
		}
	}
	if gocache == "" {
		t.Fatalf("the sandbox sets no GOCACHE: %q", p.env.sandbox)
	}
	if !slices.Contains(p.writable, realPath(gocache)) {
		t.Errorf("GOCACHE %q is not writable: %q", gocache, p.writable)
	}
	if fi, err := os.Stat(gocache); err != nil || !fi.IsDir() {
		t.Errorf("GOCACHE %q was not created: %v", gocache, err)
	}
}

func TestPlanSandboxCredentialsCanBeOptedInto(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	p, _ := testPlan(t, Sandbox{ReadCredentials: true})
	if len(p.denied) != 0 {
		t.Errorf("denied = %q, want nothing with ReadCredentials", p.denied)
	}
}

// A GOCACHE the caller sets is the one go uses, so it is the one that has to
// be writable; the sandbox does not override it.
func TestPlanSandboxKeepsACallerGOCACHE(t *testing.T) {
	mine := filepath.Join(t.TempDir(), "mine")
	p, _ := testPlan(t, Sandbox{}, WithEnv("GOCACHE="+mine))
	for _, kv := range p.env.sandbox {
		if strings.HasPrefix(kv, "GOCACHE=") {
			t.Errorf("the sandbox set %s over the caller's GOCACHE", kv)
		}
	}
	if !slices.Contains(p.writable, realPath(mine)) {
		t.Errorf("the caller's GOCACHE %q is not writable: %q", mine, p.writable)
	}
}

func TestRealPathResolvesWhatExists(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "real")
	if err := os.Mkdir(real, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Skip("no symlinks:", err)
	}
	want := filepath.Join(realPath(real), "not-yet")
	if got := realPath(filepath.Join(link, "not-yet")); got != want {
		t.Errorf("realPath = %q, want %q: the existing part is resolved, the rest kept", got, want)
	}
}

// ===== the macOS profile =====

func TestSeatbeltProfile(t *testing.T) {
	p := sandboxPlan{
		writable: []string{"/work/proj", `/odd/"quoted"\dir`},
		denied:   []string{"/home/u/.ssh"},
		network:  NetLoopback,
	}
	prof := seatbeltProfile(p)
	for _, want := range []string{
		"(allow default)",
		"(deny file-write*)",
		`(subpath "/work/proj")`,
		`(subpath "/odd/\"quoted\"\\dir")`,
		`(deny file-read* file-write*` + "\n" + `  (subpath "/home/u/.ssh")`,
		"(deny network*)",
		`(remote ip "localhost:*")`,
		"(deny appleevent-send)",
	} {
		if !strings.Contains(prof, want) {
			t.Errorf("profile is missing %s:\n%s", want, prof)
		}
	}
	// Order is the semantics: a later rule wins, so the credential deny must
	// come after the writable allow it carves out of.
	if strings.Index(prof, "(deny file-read* file-write*") < strings.Index(prof, "(allow file-write*") {
		t.Error("the credential deny precedes the writable allow and would be overridden by it")
	}

	none := seatbeltProfile(sandboxPlan{writable: []string{"/w"}, network: NetNone})
	if strings.Contains(none, `ip "localhost:*"`) {
		t.Errorf("NetNone still allows loopback:\n%s", none)
	}
	full := seatbeltProfile(sandboxPlan{writable: []string{"/w"}, network: NetFull})
	if strings.Contains(full, "(deny network*)") {
		t.Errorf("NetFull denies the network:\n%s", full)
	}
}

// TestSeatbeltProfileCompiles hands every variant of the profile to
// sandbox-exec. Inside a sandbox — CI runners and agent harnesses often are —
// sandbox-exec cannot APPLY a profile, but it compiles one first and reports a
// syntax error before it gets that far, so the check still means something.
func TestSeatbeltProfileCompiles(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("macOS only")
	}
	t.Setenv("HOME", t.TempDir())
	for _, n := range []Network{NetLoopback, NetNone, NetFull} {
		p, _ := testPlan(t, Sandbox{Network: n, DenyRead: []string{"/x/y"}})
		out, err := exec.Command("/usr/bin/sandbox-exec", "-p", seatbeltProfile(p), "/usr/bin/true").CombinedOutput()
		if err != nil && !bytes.Contains(out, []byte("sandbox_apply")) {
			t.Errorf("network %v: sandbox-exec rejected the profile: %v\n%s\n%s", n, err, out, seatbeltProfile(p))
		}
	}
}

// ===== the Linux arguments =====

func TestBwrapArgs(t *testing.T) {
	home := t.TempDir()
	if err := os.Mkdir(filepath.Join(home, ".ssh"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, ".netrc"), []byte("machine x"), 0o600); err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	p := sandboxPlan{
		writable: []string{work, "/does/not/exist"},
		denied:   []string{filepath.Join(home, ".ssh"), filepath.Join(home, ".netrc"), filepath.Join(home, ".aws")},
		network:  NetLoopback,
	}
	args := strings.Join(bwrapArgs(p), " ")
	for _, want := range []string{
		"--unshare-net",
		"--ro-bind / /",
		"--bind " + work + " " + work,
		"--tmpfs " + filepath.Join(home, ".ssh"),
		"--ro-bind /dev/null " + filepath.Join(home, ".netrc"),
		"--die-with-parent",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("bwrap args are missing %q:\n%s", want, args)
		}
	}
	for _, absent := range []string{"/does/not/exist", ".aws"} {
		if strings.Contains(args, absent) {
			t.Errorf("bwrap args mount %q, which does not exist:\n%s", absent, args)
		}
	}
	// The root must be read-only BEFORE the writable directories are bound
	// back over it, or it would cover them.
	if strings.Index(args, "--ro-bind / /") > strings.Index(args, "--bind "+work) {
		t.Errorf("the read-only root is mounted after the writable binds:\n%s", args)
	}

	if full := strings.Join(bwrapArgs(sandboxPlan{network: NetFull}), " "); strings.Contains(full, "--unshare-net") {
		t.Errorf("NetFull still unshares the network: %s", full)
	}
}

func TestParseNetwork(t *testing.T) {
	for in, want := range map[string]Network{"": NetLoopback, "loopback": NetLoopback, "none": NetNone, "FULL": NetFull} {
		if got, err := ParseNetwork(in); err != nil || got != want {
			t.Errorf("ParseNetwork(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	if _, err := ParseNetwork("some"); err == nil {
		t.Error("ParseNetwork accepted an unknown mode")
	}
}

// ===== the real thing, where it can run =====

// TestSandboxRunnerConfines runs commands through the real sandbox. It is
// skipped where none can be applied — including inside another sandbox — and
// says so, rather than passing.
func TestSandboxRunnerConfines(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.MkdirAll(filepath.Join(home, ".ssh"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, ".ssh", "id_test"), []byte("PRIVATE"), 0o600); err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	outside := t.TempDir()

	r, err := SandboxRunner(work, Sandbox{CacheDir: filepath.Join(t.TempDir(), "c"), Network: NetNone})
	if errors.Is(err, ErrNoSandbox) {
		t.Skipf("no sandbox can be applied here: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	if _, err := r.Shell(ctx, "echo ok > "+shQuote(filepath.Join(work, "in.txt"))); err != nil {
		t.Errorf("a write inside the workspace failed: %v", err)
	}
	if _, err := r.Shell(ctx, "echo no > "+shQuote(filepath.Join(outside, "out.txt"))); err == nil {
		t.Error("a write outside the workspace succeeded")
	}
	if out, _ := r.Shell(ctx, "cat "+shQuote(filepath.Join(home, ".ssh", "id_test"))); strings.Contains(out, "PRIVATE") {
		t.Error("a credential file was readable")
	}
}
