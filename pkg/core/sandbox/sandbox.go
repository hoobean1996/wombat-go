package sandbox

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"
)

// ErrUnavailable reports that [Confined] could not confine commands on this
// machine: the platform has no supported sandbox, its launcher is missing, or
// the launcher is present and refused to apply a profile — which is what
// happens when the harness is itself already running inside a sandbox.
//
// It is returned instead of an unconfined runner, never alongside one. A
// caller that asked for confinement and silently got none would be worse off
// than one that was told.
var ErrUnavailable = errors.New("sandbox: no sandbox available")

// Network is how much of the network a confined command may reach.
type Network int

const (
	// NetLoopback permits loopback and nothing else: a test that starts a
	// server on 127.0.0.1 and talks to it works, a request to the outside
	// world fails. The default.
	//
	// The platforms differ in one way that matters. On Linux the command gets
	// a network namespace of its own, so its loopback is its own too. On macOS
	// it shares the machine's loopback, and anything listening there is
	// reachable — including a forwarding proxy, which will carry a request
	// anywhere it is pointed. On a Mac that runs one, use [NetNone].
	NetLoopback Network = iota
	// NetNone permits no network at all, loopback included, on macOS. On
	// Linux it is the same isolated namespace as [NetLoopback].
	NetNone
	// NetFull leaves the network alone.
	NetFull
)

func (n Network) String() string {
	switch n {
	case NetLoopback:
		return "loopback"
	case NetNone:
		return "none"
	case NetFull:
		return "full"
	}
	return fmt.Sprintf("Network(%d)", int(n))
}

// ParseNetwork reads "loopback", "none" or "full".
func ParseNetwork(s string) (Network, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "loopback":
		return NetLoopback, nil
	case "none", "off":
		return NetNone, nil
	case "full", "on":
		return NetFull, nil
	}
	return 0, fmt.Errorf("sandbox: unknown network mode %q (want loopback, none or full)", s)
}

// Config is what a confined command may touch.
type Config struct {
	// Writable lists the directories commands may create, change and delete
	// files under. The runner's own working directory is always one of them,
	// as are the system temporary directory and CacheDir.
	Writable []string

	// Network is how much of the network commands reach. The zero value is
	// [NetLoopback].
	Network Network

	// ReadCredentials, when false — the default — makes the paths in
	// [CredentialPaths] under the home directory unreadable, and closes the
	// sockets through which a process acts with the user's authority instead
	// of its own (see [Confined]). Setting it opens both: the command can
	// then sign with the user's ssh agent, and also drive the tmux it runs in.
	ReadCredentials bool

	// DenyRead lists more paths commands may neither read nor write, nor
	// connect to if they are sockets.
	DenyRead []string

	// CacheDir is where confined toolchains keep their caches. GOCACHE points
	// at a directory under it unless [WithEnv] sets GOCACHE, and it is
	// writable. Default: wombat/sandbox under the user's cache directory.
	//
	// It is deliberately NOT the user's own GOCACHE. `go test` cannot run
	// without a writable build cache, and a cache that confined commands can
	// write and unconfined builds then read is a way out of the sandbox: an
	// entry planted under the right action id is linked into the next build
	// the user runs by hand. Keeping the confined cache separate costs one
	// cold build.
	CacheDir string
}

// CredentialPaths are the places, relative to the home directory, that hold
// credentials: ssh and gpg keys, cloud and cluster logins, registry and
// package-index tokens. A [Confined] makes them unreadable unless
// Config.ReadCredentials is set.
//
// It is a list, and lists are incomplete; this one names the places a stolen
// credential is most often taken from. A secret kept somewhere else needs a
// DenyRead entry of its own.
var CredentialPaths = []string{
	".ssh",
	".gnupg",
	".aws",
	".azure",
	".config/gcloud",
	".kube",
	".docker",
	".config/gh",
	".git-credentials",
	".netrc",
	".npmrc",
	".pypirc",
	".password-store",
	"Library/Keychains",
}

// Confined returns a Runner whose every command runs confined by the
// platform's own sandbox — sandbox-exec on macOS, bubblewrap on Linux.
//
// A confined command may write only under its working directory, the temp
// directory, Config.CacheDir and Config.Writable; it reaches only as much of
// the network as Config.Network allows; and it cannot read the credential
// paths. The environment is filtered exactly as [Unconfined] filters it.
//
// Nor can it connect to the sockets that act with the user's authority rather
// than its own, in any network mode: key agents, which sign with every key
// loaded into them; the terminal multiplexer, terminal, editor or window
// manager the harness runs inside, each of which runs a command it is sent
// outside any sandbox; the user's runtime directory and session bus, which
// reach systemd's user instance; container daemons, which are root on the
// machine. Leaving SSH_AUTH_SOCK out of the environment only hides the
// path, and the shared temp directories — writable, so open to sockets — are
// where agents and multiplexers keep theirs; it is the sandbox that makes
// them unreachable. They are found from the variables that locate them and
// the places the programs use by default, which makes this a list, as
// incomplete as lists are: a socket somewhere else needs a DenyRead entry.
// On Linux, mounts cannot hide an abstract socket — X11's, for one — and
// only the network namespace NetFull gives up keeps those out.
//
// This is the boundary the rest of the package keeps pointing at. A rooted
// [FS] confines the file tools and leaves bash alone; a permission policy
// reads the command and has to guess what it will do. A sandbox does not read
// the command at all, which is why it holds where both of those do not:
// `go test` running a test the model wrote is still unable to write outside
// the workspace or send what it read anywhere.
//
// # What it does not do
//
// It is not a jail for a hostile binary. On macOS the profile allows by
// default and denies what matters — writes, the network, credential reads, and
// the IPC routes out to the user's apps (Apple Events, LaunchServices) — so
// talking to some other system service remains possible. Reads outside the
// credential paths are allowed on both platforms, because a toolchain reads
// all over the disk; with the network closed, what is read has nowhere to go.
//
// The launcher is probed when the runner is built, so a machine that cannot
// confine — no bwrap, unprivileged user namespaces disabled, or a harness that
// is itself sandboxed — fails here with [ErrUnavailable] instead of failing on
// the model's first command.
func Confined(dir string, sb Config, opts ...Option) (Runner, error) {
	plan, err := planSandbox(dir, sb, opts)
	if err != nil {
		return nil, err
	}
	r := osRunner{dir: dir, env: plan.env}

	switch runtime.GOOS {
	case "darwin":
		const launcher = "/usr/bin/sandbox-exec"
		if _, err := os.Stat(launcher); err != nil {
			return nil, fmt.Errorf("%w: %s not found", ErrUnavailable, launcher)
		}
		profile := seatbeltProfile(plan)
		r.wrap = func(name string, args []string) (string, []string) {
			return launcher, append([]string{"-p", profile, name}, args...)
		}
	case "linux":
		launcher, err := exec.LookPath("bwrap")
		if err != nil {
			return nil, fmt.Errorf("%w: bubblewrap (bwrap) is not installed", ErrUnavailable)
		}
		prefix := bwrapArgs(plan)
		r.wrap = func(name string, args []string) (string, []string) {
			argv := append(append(append([]string(nil), prefix...), "--", name), args...)
			return launcher, argv
		}
	default:
		return nil, fmt.Errorf("%w: no sandbox on %s", ErrUnavailable, runtime.GOOS)
	}

	if err := probe(r); err != nil {
		return nil, err
	}
	return r, nil
}

// probe runs a no-op through the sandbox, so a launcher that cannot apply the
// profile is found out now.
func probe(r osRunner) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := r.Run(ctx, "/bin/sh", "-c", "exit 0")
	if err == nil {
		return nil
	}
	var ce *CommandError
	if errors.As(err, &ce) && ce.Stderr != "" {
		return fmt.Errorf("%w: the sandbox launcher refused: %s", ErrUnavailable, ce.Stderr)
	}
	return fmt.Errorf("%w: %v %s", ErrUnavailable, err, strings.TrimSpace(out))
}

// sandboxPlan is a Config with every path made absolute and resolved, and the
// defaults filled in. Resolution matters on macOS, where the sandbox matches
// the real path: /tmp is /private/tmp, and a rule naming /tmp matches nothing.
type sandboxPlan struct {
	writable []string
	denied   []string
	sockets  []string // closed to connections, directories of them included
	network  Network
	env      envConfig
}

func planSandbox(dir string, sb Config, opts []Option) (sandboxPlan, error) {
	env := newEnvConfig(opts)
	p := sandboxPlan{network: sb.Network, env: env}

	if dir == "" {
		wd, err := os.Getwd()
		if err != nil {
			return p, fmt.Errorf("sandbox: sandbox working directory: %w", err)
		}
		dir = wd
	}

	cache := sb.CacheDir
	if cache == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return p, fmt.Errorf("sandbox: sandbox cache directory: %w", err)
		}
		cache = filepath.Join(base, "wombat", "sandbox")
	}
	gocache, set := env.explicit("GOCACHE")
	if !set {
		gocache = filepath.Join(cache, "go-build")
		p.env.sandbox = append(p.env.sandbox, "GOCACHE="+gocache)
	}
	for _, d := range []string{cache, gocache} {
		if d == "" {
			continue
		}
		if err := os.MkdirAll(d, 0o755); err != nil {
			return p, fmt.Errorf("sandbox: sandbox cache directory: %w", err)
		}
	}

	writable := append([]string{dir, os.TempDir(), cache, gocache}, sb.Writable...)
	if runtime.GOOS == "darwin" {
		// The shared /tmp is a separate place from the per-user TMPDIR.
		writable = append(writable, "/private/tmp")
		// TMPDIR is /var/folders/xx/yyyy/T; its parent holds the per-user
		// cache (C) the system's own tools write to — xcrun, which cgo runs,
		// among them.
		if t := realPath(os.TempDir()); strings.HasPrefix(t, "/private/var/folders/") {
			writable = append(writable, filepath.Dir(t))
		}
	} else {
		writable = append(writable, "/tmp")
	}
	p.writable = uniquePaths(writable)

	var denied []string
	if !sb.ReadCredentials {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			for _, c := range CredentialPaths {
				denied = append(denied, filepath.Join(home, c))
			}
		}
	}
	denied = append(denied, sb.DenyRead...)
	p.denied = uniquePaths(denied)
	if !sb.ReadCredentials {
		p.sockets = uniquePaths(agentSockets())
	}
	return p, nil
}

// socketEnv names the variables whose value says where a socket that acts
// with the user's authority listens. The child never sees them — see
// safeenv — but the path is just as reachable without the variable.
var socketEnv = []string{
	// Key agents.
	"SSH_AUTH_SOCK", "GPG_AGENT_INFO",
	// What the harness may be running inside: each runs a command it is sent.
	"TMUX", "KITTY_LISTEN_ON", "WEZTERM_UNIX_SOCKET", "ALACRITTY_SOCKET",
	"NVIM", "NVIM_LISTEN_ADDRESS", "VSCODE_IPC_HOOK", "VSCODE_IPC_HOOK_CLI",
	"SWAYSOCK", "I3SOCK",
	// The session bus, and container daemons.
	"DBUS_SESSION_BUS_ADDRESS", "DOCKER_HOST", "CONTAINER_HOST",
}

// agentSockets finds the sockets, and the directories of them, a confined
// command must not connect to — the ones [Confined] describes. Where a
// program keeps one directory per user, the whole directory is listed, so a
// socket created in it later is covered too.
func agentSockets() []string {
	var out []string
	for _, name := range socketEnv {
		if p := socketPath(name, os.Getenv(name)); p != "" {
			out = append(out, p)
		}
	}

	uid := os.Getuid()
	// The shared temp directories, where OpenSSH agents, tmux, zellij and
	// the Emacs server put theirs: TMPDIR, or /tmp when it is unset.
	for _, t := range uniquePaths([]string{"/tmp", os.TempDir(), os.Getenv("TMUX_TMPDIR")}) {
		for _, n := range []string{fmt.Sprintf("tmux-%d", uid), fmt.Sprintf("zellij-%d", uid), fmt.Sprintf("emacs%d", uid)} {
			out = append(out, filepath.Join(t, n))
		}
		agents, _ := filepath.Glob(filepath.Join(t, "ssh-*"))
		out = append(out, agents...)
	}
	out = append(out, os.Getenv("SCREENDIR"), "/tmp/screens", "/var/run/screen", "/var/run/docker.sock")

	switch runtime.GOOS {
	case "darwin":
		// launchd's per-session directories: the ssh agent's Listeners
		// socket, XQuartz's display.
		for _, g := range []string{"/private/tmp/com.apple.launchd.*", "/private/var/run/com.apple.launchd.*"} {
			m, _ := filepath.Glob(g)
			out = append(out, m...)
		}
	case "linux":
		out = append(out, "/tmp/.X11-unix", "/run/screen", "/run/docker.sock",
			"/run/containerd/containerd.sock", "/run/podman/podman.sock")
		if os.Getenv("XDG_RUNTIME_DIR") == "" {
			out = append(out, fmt.Sprintf("/run/user/%d", uid))
		}
	}
	return append(out, os.Getenv("XDG_RUNTIME_DIR"))
}

// socketPath extracts the filesystem path from a socket address, spelled the
// way the variable called name spells it. It returns "" for an address that
// is not a path: an abstract socket, a TCP address, nothing at all.
func socketPath(name, v string) string {
	v = strings.TrimSpace(v)
	switch name {
	case "TMUX": // /tmp/tmux-501/default,1234,0
		v, _, _ = strings.Cut(v, ",")
	case "GPG_AGENT_INFO": // /home/u/.gnupg/S.gpg-agent:1234:1
		for range 2 {
			if i := strings.LastIndexByte(v, ':'); i >= 0 {
				v = v[:i]
			}
		}
	case "DBUS_SESSION_BUS_ADDRESS": // unix:path=/run/user/1000/bus,guid=…[;next]
		first, _, _ := strings.Cut(v, ";")
		v = ""
		if kv, ok := strings.CutPrefix(first, "unix:"); ok {
			for f := range strings.SplitSeq(kv, ",") {
				if p, ok := strings.CutPrefix(f, "path="); ok {
					v = p
				}
			}
		}
	default: // a path, unix:/path or unix:///path
		if rest, ok := strings.CutPrefix(v, "unix:"); ok {
			v = rest
			if strings.HasPrefix(v, "///") {
				v = v[2:]
			}
		}
	}
	if !filepath.IsAbs(v) {
		return ""
	}
	return v
}

// realPath makes p absolute and resolves its symlinks, as far as they exist.
// A path that does not exist yet keeps its absolute, cleaned spelling, with
// the part that does exist resolved — so a deny rule written for a directory
// that is created later still names the place it will be created.
func realPath(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		return filepath.Clean(p)
	}
	if r, err := filepath.EvalSymlinks(abs); err == nil {
		return r
	}
	parent, base := filepath.Dir(abs), filepath.Base(abs)
	if parent == abs {
		return abs
	}
	return filepath.Join(realPath(parent), base)
}

func uniquePaths(ps []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range ps {
		if p == "" {
			continue
		}
		r := realPath(p)
		if !seen[r] {
			seen[r] = true
			out = append(out, r)
		}
	}
	return out
}

// seatbeltProfile renders a plan as a macOS sandbox profile.
//
// Later rules override earlier ones, which is what the ordering relies on:
// writes are denied everywhere and then allowed back under the writable
// directories; credential paths are denied after that, so a credential inside
// a writable directory is still out of reach. The sockets come last, for the
// same reason: an agent's socket under the shared /tmp is under a directory
// whose sockets the network rules open.
func seatbeltProfile(p sandboxPlan) string {
	var b strings.Builder
	b.WriteString("(version 1)\n(allow default)\n")

	b.WriteString("(deny file-write*)\n(allow file-write*\n")
	for _, w := range p.writable {
		fmt.Fprintf(&b, "  (subpath %s)\n", sbString(w))
	}
	b.WriteString(`  (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty")` + "\n")
	b.WriteString(`  (literal "/dev/dtracehelper") (subpath "/dev/fd") (regex #"^/dev/ttys[0-9]+$"))` + "\n")

	if len(p.denied) > 0 {
		b.WriteString("(deny file-read* file-write*\n")
		for _, d := range p.denied {
			fmt.Fprintf(&b, "  (subpath %s)\n", sbString(d))
		}
		b.WriteString(")\n")
	}

	// The routes out through the user's own apps: Apple Events drive another
	// application, LaunchServices opens a URL in the browser. Either one sends
	// data past a closed network.
	b.WriteString("(deny appleevent-send)\n")
	b.WriteString(`(deny mach-lookup (global-name "com.apple.coreservices.launchservicesd") (global-name-prefix "com.apple.lsd."))` + "\n")

	if p.network != NetFull {
		b.WriteString("(deny network*)\n")
		if p.network == NetLoopback {
			b.WriteString(`(allow network-bind network-inbound (local ip "localhost:*"))` + "\n")
			b.WriteString(`(allow network-outbound (remote ip "localhost:*"))` + "\n")
		}
		// Unix sockets under a writable directory are a test talking to
		// itself. Others — the DNS resolver's, a container daemon's — are a
		// way out, and stay closed.
		for _, w := range p.writable {
			fmt.Fprintf(&b, "(allow network-bind network-inbound (local unix-socket (subpath %s)))\n", sbString(w))
			fmt.Fprintf(&b, "(allow network-outbound (remote unix-socket (subpath %s)))\n", sbString(w))
		}
	}

	// In every network mode: a socket that acts for the user, and anything
	// under a denied path — gpg-agent listens in ~/.gnupg.
	for _, s := range append(slices.Clip(p.sockets), p.denied...) {
		fmt.Fprintf(&b, "(deny network-outbound (remote unix-socket (subpath %s)))\n", sbString(s))
	}
	return b.String()
}

// sbString quotes s as a sandbox-profile string literal.
func sbString(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s) + `"`
}

// bwrapArgs renders a plan as bubblewrap options, everything before the "--"
// that precedes the command.
//
// The root is bound read-only and the writable directories are bound back
// over it. A credential or socket directory is hidden under an empty tmpfs,
// and a credential file or a socket under /dev/null, which refuses a
// connection; none is mounted if it does not exist, because bwrap cannot mount
// over nothing. Every network mode but NetFull gets a namespace of its own,
// whose loopback belongs to the command alone.
//
// A mount covers whatever was mounted at or under its path before it, so the
// order is the semantics. Writable directories and sockets go shallowest
// first, which makes the more specific of two nested paths win: an agent's
// socket under the writable /tmp is hidden, a writable TMPDIR inside the
// hidden runtime directory is bound back. Denied paths go last and win
// outright, as they do in the macOS profile.
func bwrapArgs(p sandboxPlan) []string {
	args := []string{
		"--die-with-parent",
		// A new session: a command cannot push keystrokes into the
		// terminal the harness runs in (TIOCSTI).
		"--new-session",
		"--unshare-pid",
	}
	if p.network != NetFull {
		args = append(args, "--unshare-net")
	}
	args = append(args, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc")

	type mount struct {
		path string
		hide bool
	}
	var ms []mount
	for _, w := range p.writable {
		ms = append(ms, mount{w, false})
	}
	for _, s := range p.sockets {
		ms = append(ms, mount{s, true}) // after the writable ones: at the same path, hiding wins
	}
	slices.SortStableFunc(ms, func(a, b mount) int { return cmp.Compare(pathDepth(a.path), pathDepth(b.path)) })
	for _, m := range ms {
		if m.hide {
			args = append(args, hideArgs(m.path)...)
		} else if _, err := os.Stat(m.path); err == nil {
			args = append(args, "--bind", m.path, m.path)
		}
	}
	for _, d := range p.denied {
		args = append(args, hideArgs(d)...)
	}
	return args
}

// hideArgs mounts over path: an empty tmpfs on a directory, /dev/null on
// anything else. Nothing, if there is nothing there.
func hideArgs(path string) []string {
	fi, err := os.Stat(path)
	switch {
	case err != nil:
		return nil
	case fi.IsDir():
		return []string{"--tmpfs", path}
	default:
		return []string{"--ro-bind", "/dev/null", path}
	}
}

func pathDepth(p string) int { return strings.Count(filepath.Clean(p), string(filepath.Separator)) }
