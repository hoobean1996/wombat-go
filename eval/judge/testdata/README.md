# Labelled trajectories

A matched pair, produced by the same agent on the same task with the same
requirement, and differing in exactly one thing: whether the claim the answer
makes is true.

Both answers assert that the state file lives beside the compiled binary rather
than in the working directory. Ground truth was established by hand, by running
each binary from a different directory and looking at what it read:

| fixture | claim | discriminating observation in the trajectory | truth |
|---|---|---|---|
| `claim-false.session.json` | "todos.json … next to the binary" | none — every check ran inside the workspace | **false**, the path is relative to the working directory |
| `claim-true.session.json` | "todos.json is stored in the same directory as the compiled binary" | yes — one call runs the binary from `/tmp` and confirms no file appears there | **true**, resolved via `os.Executable` |

The pair exists because a single mislabelled trajectory cannot tell a judge that
detects the defect apart from a judge that distrusts every confident answer.
Separating these two is the test; scoring the first one badly is not.

The lesson they encode is about EVIDENCE, not about confidence. In the false
case a model judge scored faithfulness 5 at confidence 1.00 and cited the
agent's own functional test — a test that ran inside the workspace and therefore
could not have come out differently had the claim been false. The question a
judge has to ask is not "is there evidence" but "could this evidence have
distinguished the claim from its negation".
