package builtin

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/internal/fspath"
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
	t.Setenv("SSH_AUTH_SOCK", "/agents/ssh.sock")
	p, _ := testPlan(t, Sandbox{ReadCredentials: true})
	if len(p.denied) != 0 || len(p.sockets) != 0 {
		t.Errorf("denied = %q, sockets = %q; want neither with ReadCredentials", p.denied, p.sockets)
	}
}

// TestPlanSandboxClosesAgentSockets: leaving SSH_AUTH_SOCK out of the child's
// environment hid the path and nothing else. The agent's socket sat under the
// shared temp directory, which is writable and so open to sockets, and the
// child could list that directory as well as the harness could.
func TestPlanSandboxClosesAgentSockets(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix socket paths")
	}
	t.Setenv("HOME", t.TempDir())
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	agentDir := filepath.Join(tmp, "ssh-AbCd1234") // where ssh-agent puts one it starts
	if err := os.Mkdir(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	rt := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", rt)
	t.Setenv("SSH_AUTH_SOCK", "/agents/ssh.sock")
	t.Setenv("TMUX", "/mux/tmux-1/default,4242,0")
	t.Setenv("DOCKER_HOST", "unix:///daemons/docker.sock")
	t.Setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/bus/session,guid=1f2e")
	t.Setenv("NVIM", "127.0.0.1:6666") // an address, not a path

	p, _ := testPlan(t, Sandbox{})
	for _, want := range []string{
		"/agents/ssh.sock", "/mux/tmux-1/default", "/daemons/docker.sock", "/bus/session",
		realPath(rt), realPath(agentDir),
		filepath.Join(realPath(tmp), fmt.Sprintf("tmux-%d", os.Getuid())),
	} {
		if !slices.Contains(p.sockets, want) {
			t.Errorf("sockets %q is missing %q", p.sockets, want)
		}
	}
	for _, s := range p.sockets {
		if strings.Contains(s, "6666") {
			t.Errorf("a TCP address was taken for a path: %q", s)
		}
	}
}

func TestSocketPath(t *testing.T) {
	for _, tc := range []struct{ name, v, want string }{
		{"SSH_AUTH_SOCK", "/tmp/ssh-x/agent.1", "/tmp/ssh-x/agent.1"},
		{"SSH_AUTH_SOCK", "", ""},
		{"SSH_AUTH_SOCK", "relative/agent", ""},
		{"TMUX", "/tmp/tmux-501/default,1234,0", "/tmp/tmux-501/default"},
		{"GPG_AGENT_INFO", "/home/u/.gnupg/S.gpg-agent:1234:1", "/home/u/.gnupg/S.gpg-agent"},
		{"DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus", "/run/user/1000/bus"},
		{"DBUS_SESSION_BUS_ADDRESS", "unix:guid=ab,path=/run/user/1000/bus;tcp:host=x", "/run/user/1000/bus"},
		{"DBUS_SESSION_BUS_ADDRESS", "unix:abstract=/tmp/dbus-x,guid=ab", ""},
		{"DOCKER_HOST", "unix:///var/run/docker.sock", "/var/run/docker.sock"},
		{"DOCKER_HOST", "tcp://127.0.0.1:2375", ""},
		{"KITTY_LISTEN_ON", "unix:/tmp/kitty", "/tmp/kitty"},
		{"KITTY_LISTEN_ON", "unix:@kitty", ""},
		{"NVIM", "127.0.0.1:6666", ""},
	} {
		if runtime.GOOS == "windows" {
			t.Skip("unix socket paths")
		}
		if got := socketPath(tc.name, tc.v); got != tc.want {
			t.Errorf("socketPath(%s=%q) = %q, want %q", tc.name, tc.v, got, tc.want)
		}
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

	// The sockets are closed in every network mode — NetFull opens the
	// network, not the user's agents — and after the rules that open the
	// sockets under a writable directory, which would otherwise win.
	for _, n := range []Network{NetLoopback, NetNone, NetFull} {
		prof := seatbeltProfile(sandboxPlan{
			writable: []string{"/private/tmp"},
			sockets:  []string{"/private/tmp/ssh-x"},
			denied:   []string{"/home/u/.gnupg"},
			network:  n,
		})
		for _, want := range []string{
			`(deny network-outbound (remote unix-socket (subpath "/private/tmp/ssh-x")))`,
			`(deny network-outbound (remote unix-socket (subpath "/home/u/.gnupg")))`,
		} {
			i := strings.Index(prof, want)
			if i < 0 {
				t.Errorf("network %v: the profile is missing %s:\n%s", n, want, prof)
			} else if strings.LastIndex(prof, "(allow network-outbound") > i {
				t.Errorf("network %v: %s precedes an allow that overrides it:\n%s", n, want, prof)
			}
		}
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

// TestBwrapArgsHideSockets: a mount covers what was mounted under it before,
// so the order of the arguments decides which of two nested paths wins.
func TestBwrapArgsHideSockets(t *testing.T) {
	base := t.TempDir()
	dir := func(rel string) string {
		p := filepath.Join(base, rel)
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Fatal(err)
		}
		return p
	}
	tmp := dir("tmp")          // writable
	agents := dir("tmp/ssh-x") // a socket directory under it: hidden
	rt := dir("run")           // the runtime directory: hidden
	rtTmp := dir("run/tmp")    // a writable TMPDIR inside it: bound back
	sock := filepath.Join(rtTmp, "agent.sock")
	if err := os.WriteFile(sock, nil, 0o600); err != nil { // stands in for a socket under that
		t.Fatal(err)
	}
	same := dir("same") // both writable and a socket directory: hidden
	p := sandboxPlan{
		writable: []string{tmp, rtTmp, same},
		sockets:  []string{sock, agents, rt, same, filepath.Join(base, "gone")},
		network:  NetLoopback,
	}
	args := " " + strings.Join(bwrapArgs(p), " ") + " "
	at := func(arg string) int {
		i := strings.Index(args, " "+arg+" ")
		if i < 0 {
			t.Errorf("bwrap args are missing %q:\n%s", arg, args)
		}
		return i
	}
	for _, order := range [][]string{
		{"--bind " + tmp + " " + tmp, "--tmpfs " + agents},
		{"--tmpfs " + rt, "--bind " + rtTmp + " " + rtTmp, "--ro-bind /dev/null " + sock},
		{"--bind " + same + " " + same, "--tmpfs " + same},
	} {
		for i := 1; i < len(order); i++ {
			if a, b := at(order[i-1]), at(order[i]); a >= 0 && b >= 0 && a > b {
				t.Errorf("%q comes after %q, so it is covered by it:\n%s", order[i-1], order[i], args)
			}
		}
	}
	if strings.Contains(args, "gone") {
		t.Errorf("bwrap args mount a path that does not exist:\n%s", args)
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
	sb := Sandbox{CacheDir: filepath.Join(t.TempDir(), "c"), Network: NetNone}

	// The ssh agent's socket, somewhere only SSH_AUTH_SOCK says: under the
	// temp directory, whose sockets are open to a test talking to itself.
	// Short paths, because a socket's is capped at about 100 bytes.
	agent, agentConns := listen(t, "ag")
	t.Setenv("SSH_AUTH_SOCK", agent)
	own, ownConns := listen(t, "ok")

	r, err := SandboxRunner(work, sb)
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
	if out, _ := r.Shell(ctx, "cat "+shQuote(filepath.Join(home, ".ssh", "id_test"))); strings.Contains(out, "PRIVATE") {
		t.Error("a credential file was readable")
	}

	// Outside means outside every writable directory, and the temp
	// directory is one of them — the package directory is not, unless the
	// checkout itself lives somewhere writable.
	outside, err := os.MkdirTemp(".", ".sandbox-outside-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(outside) })
	outside = realPath(outside)
	plan, err := planSandbox(work, sb, nil)
	if err != nil {
		t.Fatal(err)
	}
	if slices.ContainsFunc(plan.writable, func(w string) bool { return fspath.Within(w, outside) }) {
		t.Logf("%s is under a writable directory; not checking a write outside them", outside)
	} else {
		target := filepath.Join(outside, "out.txt")
		if _, err := r.Shell(ctx, "echo no > "+shQuote(target)); err == nil {
			t.Error("a write outside the writable directories succeeded")
		}
		if _, err := os.Stat(target); err == nil {
			t.Error("a write outside the writable directories left a file")
		}
	}

	nc, err := exec.LookPath("nc")
	if err != nil {
		t.Log("no nc; not checking the sockets")
		return
	}
	dial := func(sock string) { _, _ = r.Shell(ctx, "echo hi | "+shQuote(nc)+" -U -w 2 "+shQuote(sock)) }
	dial(own)
	if !eventually(func() bool { return ownConns.Load() > 0 }) {
		t.Log("nc reached no socket at all, not even one the sandbox allows; not checking the agent's")
		return
	}
	dial(agent)
	if eventually(func() bool { return agentConns.Load() > 0 }) {
		t.Error("a confined command connected to the ssh agent's socket")
	}
}

// listen accepts connections on a unix socket in a new directory under the
// temp directory, and counts them.
func listen(t *testing.T, prefix string) (string, *atomic.Int32) {
	t.Helper()
	d, err := os.MkdirTemp("", prefix)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(d) })
	path := filepath.Join(d, "s")
	l, err := net.Listen("unix", path)
	if err != nil {
		t.Skipf("cannot listen on %s: %v", path, err)
	}
	t.Cleanup(func() { _ = l.Close() })
	var n atomic.Int32
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			n.Add(1)
			_ = c.Close()
		}
	}()
	return path, &n
}

func eventually(cond func() bool) bool {
	for deadline := time.Now().Add(time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		if cond() {
			return true
		}
	}
	return cond()
}
