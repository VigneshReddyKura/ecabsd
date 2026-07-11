#!/bin/bash
# ECABSD — Git setup script.
# Configures user identity locally and installs git hooks to enforce quality.

echo "=== ECABSD Git Setup ==="

# 1. Setup local identity if not set
CURRENT_NAME=$(git config user.name)
CURRENT_EMAIL=$(git config user.email)

echo "Current local Git user.name:  '$CURRENT_NAME'"
echo "Current local Git user.email: '$CURRENT_EMAIL'"

read -p "Do you want to update your Git identity for this project? (y/N): " UPDATE_ID
if [[ "$UPDATE_ID" =~ ^[Yy]$ ]]; then
    read -p "Enter your full name (for commits): " NEW_NAME
    read -p "Enter your email (matching GitHub account): " NEW_EMAIL
    if [ ! -z "$NEW_NAME" ] && [ ! -z "$NEW_EMAIL" ]; then
        git config user.name "$NEW_NAME"
        git config user.email "$NEW_EMAIL"
        echo "Updated Git identity successfully!"
    else
        echo "Name or email cannot be empty. Identity not updated."
    fi
fi

# 2. Setup commit-msg hook for Conventional Commits
HOOKS_DIR=".git/hooks"
COMMIT_MSG_HOOK="$HOOKS_DIR/commit-msg"

echo "Installing Conventional Commits hook..."
mkdir -p "$HOOKS_DIR"

cat << 'EOF' > "$COMMIT_MSG_HOOK"
#!/bin/bash
# Validate commit message format

commit_regex='^(merge|feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|release)(\([a-zA-Z0-9_-]+\))?: .+'
commit_msg=$(cat "$1")

if [[ ! "$commit_msg" =~ $commit_regex ]]; then
    echo "ERROR: Commit message does not follow Conventional Commits format."
    echo "Example: 'feat: add docker support' or 'fix(web): resolve crash'"
    echo "Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert, release"
    exit 1
fi
EOF

chmod +x "$COMMIT_MSG_HOOK"
echo "Commit message hook installed successfully!"
echo "=== Setup Complete ==="
