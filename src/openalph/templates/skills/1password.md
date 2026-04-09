<!-- Shipped with OpenAlph. Customize for your setup. -->
# 1Password CLI

Use 1Password CLI (`op`) for secret retrieval and injection. Each agent has its own service account token and vault — check your ENVIRONMENT.md for your specific token path and vault name.

## Usage Pattern

Always export the service account token before running `op` commands. The token path is agent-specific — find yours in your ENVIRONMENT.md:

```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op <command>
```

## Common Commands

### Verify access
```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op whoami
```

### List vaults
```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op vault list
```

### Read a secret
```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op item get "item-name" --vault <your-vault> --fields password
```

### Read with op:// URI (preferred)
```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op read "op://<your-vault>/item-name/field"
```

### Run a command with injected secrets
```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
op run --env-file <(echo 'API_KEY=op://<your-vault>/item/field') -- <command>
```

### Inject secrets into a template
```bash
export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.config/op/service-account-token)
cat template.txt | op inject
```

## Guardrails

- Never paste secrets into logs, chat, or code.
- Prefer `op run` / `op inject` over writing secrets to disk.
- Never commit the service account token or any credentials to git.
- Never output private keys, mnemonics, seed phrases, or raw API keys to chat.
- Never include secrets in prompts sent to any inference API.
