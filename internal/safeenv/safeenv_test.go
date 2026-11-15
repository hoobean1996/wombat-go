package safeenv

import (
	"slices"
	"testing"
)

func TestFilterKeepsWhatABuildNeedsAndNoCredential(t *testing.T) {
	environ := []string{
		"PATH=/usr/bin",
		"HOME=/home/u",
		"LC_ALL=C",
		"GOCACHE=/c",
		"CGO_ENABLED=0",
		"OPENAI_API_KEY=sk-1",
		"ANTHROPIC_API_KEY=sk-2",
		"ANTHROPIC_CUSTOM_HEADERS=x-auth: t",
		"OPENAI_EXTRA_HEADERS=x-auth: t",
		"GOOGLE_API_KEY=g",
		"GOOGLE_APPLICATION_CREDENTIALS=/k.json",
		"AWS_SECRET_ACCESS_KEY=a",
		"GITHUB_TOKEN=gh",
		"SSH_AUTH_SOCK=/tmp/agent",
		"=C:=C:\\dir",
	}
	got := Filter(environ, DefaultAllow)
	want := []string{"PATH=/usr/bin", "HOME=/home/u", "LC_ALL=C", "GOCACHE=/c", "CGO_ENABLED=0"}
	if !slices.Equal(got, want) {
		t.Errorf("Filter = %q\nwant     %q", got, want)
	}
}

func TestAllowedPrefixAndExact(t *testing.T) {
	allow := []string{"FOO", "BAR_*"}
	for name, want := range map[string]bool{
		"FOO":     true,
		"FOOD":    false, // exact, not a prefix
		"BAR_":    true,
		"BAR_BAZ": true,
		"BAR":     false,
		"":        false,
	} {
		if got := Allowed(name, allow); got != want {
			t.Errorf("Allowed(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestOverlayReplacesAndAppends(t *testing.T) {
	base := []string{"A=1", "B=2", "C=3"}
	got := Overlay(base, []string{"B=20", "D=4"})
	want := []string{"A=1", "C=3", "B=20", "D=4"}
	if !slices.Equal(got, want) {
		t.Errorf("Overlay = %q, want %q", got, want)
	}
	if !slices.Equal(base, []string{"A=1", "B=2", "C=3"}) {
		t.Errorf("Overlay modified its base: %q", base)
	}
	if got := Overlay(base, nil); !slices.Equal(got, base) {
		t.Errorf("Overlay with nothing to set = %q, want the base unchanged", got)
	}
}

func TestEnvironReadsTheProcessNow(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("GOFLAGS", "-mod=mod")
	env := Environ()
	if slices.Contains(env, "OPENAI_API_KEY=sk-test") {
		t.Error("Environ passed the provider key through")
	}
	if !slices.Contains(env, "GOFLAGS=-mod=mod") {
		t.Error("Environ dropped a variable set after the process started")
	}
}
