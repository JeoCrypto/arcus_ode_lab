# Arcus clients: website terminal vs SSH

## Website (`https://www.augustalabs.ai/arcus`)

The Arcus prize page is a **Framer** site. The “terminal” is **not** a browser extension and **not** a live SSH session in the tab.

It is an **iframe** embedded from:

```text
https://arcus-tui.vercel.app/
```

Relevant HTML (from the published page):

```html
<iframe src="https://arcus-tui.vercel.app/" sandbox="allow-same-origin allow-scripts allow-..." />
```

### What `arcus-tui.vercel.app` actually is

- A **static fake terminal** (HTML + CSS + vanilla JS).
- Commands are a hardcoded `COMMANDS` map: `help`, `start`, `arcus`, `prize`, `submission`, `write-up`, `contact`, `whois augustalabs`, `clear`.
- The only “real” action is copying `ssh augustalabs.ai` to the clipboard.
- **No** WebSocket, **no** xterm.js, **no** `fetch()` to a flag API in the bundle we inspected.
- `SUBMISSIONS = 0` is a constant in JS (not live stats unless they redeploy).

Your poem + `flag: Hup-la...` + `checking...` screenshot is **not** produced by this embed (no poem / flag strings in that app). That UI comes from the **real SSH Bubble Tea app** (or a retired `/ode` page).

### How to verify in the browser

1. Open DevTools on `/arcus` → **Elements** → find the iframe `src`.
2. Open `https://arcus-tui.vercel.app/` in its own tab → **Network** (should stay empty on typing).
3. **Application** → Local Storage (only Framer/editor keys on parent; fake terminal has no flag cache).

## SSH (`ssh augustalabs.ai`)

This is the **real** challenge client:

- Server: Go SSH (`SSH-2.0-Go`), PTY required.
- UI: Charm **Wish** + **Bubble Tea** (inferred; source not public).
- Our automation (`submit_flag.expect`, `try_ssh_flags.expect`) goes **straight to Ode Triunfal** (one menu down + Enter, or `start` → `arcus`).

### Did we explore “disk folders”?

**No — not systematically.** Scripts assume a single menu path to the `flag:` prompt. We did **not** yet:

- Select menu row 0 / 2 / 3 (other trials or folders).
- Run `ls`, `cd`, `dir`, `help` at the root menu or inside Ode.
- Walk `ARCUS_MENU_DOWN=0..3` and log each screen.

Use:

```bash
expect scripts/explore_ssh_menu.expect
ARCUS_MENU_DOWN=0 expect scripts/explore_ssh_menu.expect
ARCUS_NAV=commands expect scripts/explore_ssh_menu.expect
```

Read the transcript for filesystem metaphors (`cd`, folders, multiple challenges). If the menu is only a list of trials, “folders” may be UI copy, not a real disk.

## `/ode` URL

`https://augustalabs.ai/ode` currently **302 redirects** to the GitHub release (`ode.pt` only). It is unrelated to the Vercel iframe terminal.

## Summary

| Surface | Real? | Flag validation | “Cache” |
|--------|-------|-----------------|--------|
| Framer `/arcus` iframe | Marketing shell | No | Clipboard + static text |
| `arcus-tui.vercel.app` | Fake terminal | No | None |
| `ssh augustalabs.ai` | Real TUI | Yes (server-side) | Server session state |
| `ode.pt` download | Model weights | Used by server (likely) | N/A |

Wireshark on SSH still only shows encrypted port 22. Use **session transcripts** (`script`, `explore_ssh_menu.expect`) to map navigation, not packet payloads.