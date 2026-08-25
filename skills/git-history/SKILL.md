---
name: git-history
description: Reading a repository's history — finding when a line changed, why a decision was made, and who to ask. Load before any question about the past rather than the present state of the code.
---

# Reading git history

The tools `git_log` and `git_show` are gated behind this skill. They are
available now.

## Finding when a line changed

`git log -L <start>,<end>:<file>` follows a range through renames. Prefer it
over `git blame` when you want the sequence of changes rather than the single
last author.

For a symbol rather than a line range:

```
git log -S'funcName' --oneline -- path/
```

`-S` finds commits where the *count* of the string changed — introductions and
deletions. `-G` matches the diff text itself and is noisier.

## Finding why

A commit message answers "what". The surrounding commits answer "why". After
locating a commit, read its neighbours:

```
git log --oneline -5 <sha>
git show <sha> --stat
```

A one-line commit body in a repository that otherwise writes paragraphs is
itself a signal: it usually means the change was mechanical or urgent.

## What not to conclude

- A file with many commits is not necessarily unstable; it may just be a
  central file everyone touches.
- The last person to touch a line is often not its author — formatting sweeps
  and renames rewrite authorship. Check whether the commit is a bulk change
  before naming anyone.
