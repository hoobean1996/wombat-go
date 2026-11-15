import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

// wombatd, for `npm run dev`: the dev server forwards the API to it.
const wombatd = process.env.WOMBATD_URL || 'http://127.0.0.1:8080'

// ui/embed.go embeds dist/ into wombatd, and go:embed refuses a pattern that
// matches nothing — so dist/.gitkeep is committed, and put back after every
// build because Vite empties dist/ first. Without it, `go build ./...` fails on
// a checkout where nobody has built the UI.
const keepPlaceholder = (): Plugin => ({
  name: 'keep-dist-placeholder',
  apply: 'build',
  closeBundle() { writeFileSync(join(import.meta.dirname, 'dist', '.gitkeep'), '') },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), keepPlaceholder()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      // changeOrigin sends wombatd its own host name, which its DNS-rebinding
      // check expects; the event stream passes through as a long response.
      '/api': { target: wombatd, changeOrigin: true },
      '/metrics': { target: wombatd, changeOrigin: true },
    },
  },
})
