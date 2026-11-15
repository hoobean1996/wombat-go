package daemon_test

import (
	"testing"

	"wombat/pkg/daemon"
	"wombat/pkg/daemon/storetest"
)

func TestMemoryStore(t *testing.T) {
	storetest.Run(t, func(*testing.T) daemon.Store { return daemon.MemoryStore() })
}
