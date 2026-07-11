# Professional Git Workflow Guide

This guide details the Git practices and setup for the ECABSD project to ensure that your commit history remains clean, and your GitHub contributions (green profile activity blocks) are accurately tracked and never lost.

---

## 1. Identity & Contribution Attribution

GitHub attributes contributions (green squares) to your profile based on the **email address** configured in your local Git repository. If this email does not match your GitHub account email, your contributions will not show up on GitHub!

### Check Your Configured Email
Run this in your terminal:
```bash
git config user.name
git config user.email
```

### Set Correct Identity Locally
To configure your identity for this project:
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```
*(Ensure this email is added to your GitHub profile at: Settings -> Emails)*

---

## 2. Managing Multiple Remotes

This project uses two remote repositories:
1. **origin** (pointing to your personal repo: `https://github.com/amanigreeva/ECABSD.git`)
2. **vignesh** (pointing to the shared repo: `https://github.com/VigneshReddyKura/ecabsd.git`)

### View Registered Remotes
```bash
git remote -v
```

### Syncing Pushes to Both Remotes
Whenever you want to push changes to both remotes, run:
```bash
# Push main branch to origin
git push origin main

# Push main branch to vignesh
git push vignesh main
```

---

## 3. Preserving Commits and Contributions

### Avoid Destructive Rewrites
- **`git reset --hard` / `git push --force`**: Only use force pushing when deliberately syncing split histories (like our recent restoration). Avoid force-pushing on shared branches as it overwrites others' history.
- **Squash Merging**: If you merge a Pull Request on GitHub using "Squash and merge", all your individual commits are squashed into a single commit. While this keeps the main branch history clean, it reduces your commit count to 1. If you want to keep every individual commit, use **Rebase and merge** or **Create a merge commit**.

---

## 4. Conventional Commits

Using structured commit messages keeps history organized and allows for automatic changelog generation.

### Commit Format
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: A new feature (e.g. `feat: add explainability visualization`)
- **fix**: A bug fix (e.g. `fix: resolve PDB file parser crash`)
- **docs**: Documentation changes only (e.g. `docs: update setup guide`)
- **style**: Changes that do not affect the meaning of the code (e.g. formatting)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Build process, tool updates, or auxiliary library updates (e.g. `chore: update requirements.txt`)
