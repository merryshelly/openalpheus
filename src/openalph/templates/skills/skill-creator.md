<!-- Shipped with OpenAlph. Customize for your setup. -->
# Skill Creator

Use when designing, structuring, or creating skills for OpenAlph. Skills are modular knowledge packages that extend Merry's capabilities by providing specialized workflows, tool integrations, and domain expertise.

## What Makes a Good Skill

A skill transforms general capability into specific competence. It should:

1. **Solve a real problem** - Capture knowledge that would otherwise be re-derived each time
2. **Be self-contained** - Everything needed, in one file (or clearly linked)
3. **Respect context** - Only include what future-you won't already know
4. **Set appropriate guardrails** - Match specificity to task fragility

## Skill Structure

Skills are flat `.md` files in `workspace/skills/`. No subdirectories, no YAML frontmatter, no special packaging.

```
workspace/skills/
├── skill-creator.md
├── browser-automation.md
├── alerts-management.md
└── ...
```

### Anatomy of a Skill File

```markdown
# Skill Name

Brief description of what this skill provides and when to use it.

## Quick Reference

- Key fact 1
- Key fact 2
- Command pattern: `example`

## Section: Specific Topic

Detailed guidance for a specific aspect...

## Section: Another Topic

More detailed guidance...
```

### Naming Conventions

- **Filename**: lowercase, hyphens, no spaces (`skill-creator.md`, `cost-tracking.md`)
- **Title**: Clear, descriptive (`# Cost Tracking`)
- **Length**: Under 500 lines. Over that, split into multiple skills or move details to `memory/`

## Degrees of Freedom

Match specificity to the task:

| Freedom Level | Use When | Example |
|--------------|----------|---------|
| **High** (text instructions) | Multiple valid approaches, context-dependent | "Use judgment for retry logic" |
| **Medium** (pseudocode/scripts with params) | Preferred pattern exists, some variation OK | Template script with configurable variables |
| **Low** (specific scripts, few params) | Fragile operations, consistency critical | Exact curl command with auth token path |

## Writing Guidelines

### Start With the Essential

Default assumption: Merry is already competent. Only add what she won't know:

- **Tool locations** - Full paths to binaries (`/srv/openalph/shared/bin/bd`)
- **Auth patterns** - Exact env var names, token file paths
- **Gotchas** - The thing that always breaks ("API returns cents, not dollars")
- **Decision trees** - When to use A vs B

### Omit the Obvious

Don't explain:
- How bash works
- What JSON is
- Basic git operations
- That errors should be handled

### Concrete Over Abstract

**Bad:** "Use the appropriate tool to query costs"
**Good:** "Query with: `curl -H 'x-api-key: $KEY' https://api.anthropic.com/v1/organizations/cost_report`"

### Progressive Disclosure

Structure so the most common case is first:

1. Quick reference (the 80% case)
2. Common variations (the next 15%)
3. Edge cases and details (the last 5%)

## Skill Discovery

Merry sees skill names listed in her system prompt. She reads them on demand via `file_read`. This means:

- **Names matter** - A clear name triggers the right skill
- **First paragraph matters** - That's what she sees first when reading
- **Self-contained matters** - She may read this in isolation, without other context

## When to Create a Skill

Create a skill when you find yourself:
- Looking up the same command for the 3rd time
- Explaining the same workflow pattern repeatedly
- Working with a specific tool/API that has quirks
- Doing tasks that require domain knowledge (validator ops, 3D printing, etc.)

Don't create a skill for:
- One-off tasks
- Information easily found in official docs
- Things that change constantly (version-specific install steps)

## Skill Template

```markdown
# Skill Name

One-line description of what this skill provides.

## Quick Reference

- Key tool: `/path/to/bin`
- Config file: `/path/to/config`
- Auth: `export TOKEN=$(cat /path/to/token)`

## Usage Patterns

### Common Task A

```bash
# Step 1
command one

# Step 2
command two
```

### Common Task B

```bash
# Different approach
command three
```

## Gotchas

- API returns cents, divide by 100 for dollars
- Token expires every 24 hours
- Rate limit: 100 req/min

## Related

- Other skill to check for X
- Memory file for detailed reference
```

## Maintenance

Skills evolve. When you hit friction:

1. **Update the skill** - Add the missing context
2. **Split if bloated** - Over 500 lines suggests two skills
3. **Delete if obsolete** - Remove skills that no longer apply

The goal is accurate, useful context — not comprehensive documentation.
