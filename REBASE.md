# Git Rebase Guidelines

## Overview

Git rebase is a powerful command that allows you to integrate changes from one branch into another by rewriting the commit history. Unlike `git merge`, which creates a new commit to combine changes, rebase reorganizes the commit sequence to maintain a linear project history.

## When to Use Rebase

- When you want to maintain a clean, linear project history
- Before merging a feature branch into the main branch
- When you need to incorporate the latest changes from the main branch into your feature branch

## Basic Rebase Workflow

### 1. Create a Backup (Recommended)

Before starting a rebase, it's recommended to create a backup of your current branch:

```bash
git branch backup_mybranch
```

### 2. Start the Rebase

From your feature branch, initiate the rebase:

```bash
git rebase main
```

### 3. Resolve Conflicts

During rebase, you might encounter conflicts. For each conflict:

1. Examine the conflicts in the affected files
2. Resolve the conflicts manually in your code editor
3. Stage the resolved files:
   ```bash
   git add <resolved-files>
   ```
4. Continue the rebase:
   ```bash
   git rebase --continue
   ```
5. Repeat until all conflicts are resolved

### 4. Monitor Progress

Use `git status` frequently during the rebase process to track your progress and understand what actions are needed.

## Important Commands

- `git rebase --abort`: Cancel the rebase and return to the state before rebase started
- `git rebase --skip`: Skip the current commit and continue with the next one
- `git rebase -i main`: Start an interactive rebase, allowing you to modify commits

## Best Practices

1. Always create a backup branch before rebasing
2. Never rebase commits that have been pushed to public/shared branches
3. Keep commits atomic and well-documented
4. Use `git status` frequently to monitor the rebase progress
5. If you're unsure about the rebase process, don't hesitate to use `git rebase --abort`

## Troubleshooting

If you encounter issues during rebase:

1. Check the current status with `git status`
2. Use `git rebase --abort` to start over if needed
3. Consult the backup branch if you need to reference the original state
4. Ensure all conflicts are properly resolved before continuing

## Notes

- Rebase rewrites commit history, so use it cautiously on shared branches
- Always communicate with your team when rebasing shared branches
- Consider using `git merge` instead if you're unsure about rebase implications


## Quick Hack Alternative

While the standard rebase process is recommended, there are situations where a complex rebase might be simplified using an alternative approach. This method can be useful when dealing with numerous conflicts, but should be used with caution.

### Warning ⚠️
- This approach essentially recreates your branch rather than preserving history
- Force pushing will overwrite remote history
- Only use this method on branches that aren't shared with other developers
- Make sure you have a backup of your work before proceeding

### Alternative Workflow

1. Create a new branch from main:
   ```bash
   git checkout main
   git checkout -b copy_main
   ```

2. Identify changed files between your working branch and main:
   ```bash
   git diff dev --name-only
   ```
   This command lists all files that differ between branches.

3. Copy changed files to the new branch:
   ```bash
   # For each file in the diff list:
   git checkout dev file_name
   ```
   Then stage and commit the changes:
   ```bash
   git add .
   git commit -m "Recreated branch with latest main"
   ```

4. Rename branches:
   ```bash
   git branch -m dev dev_old      # Rename current dev to dev_old
   git branch -m copy_main dev    # Rename copy_main to dev
   ```

5. Update remote branch:
   ```bash
   git push -f origin dev
   ```

   **CRITICAL WARNING**: Force pushing (`-f`) overwrites remote history. Only use this if:
   - You are the only one working on this branch
   - You have confirmed no one else has unpushed changes
   - You have a backup of the original branch
   - Your team is aware of this operation

### When to Use This Approach

Consider this method when:
- The standard rebase process is resulting in numerous complex conflicts
- You're working on a personal feature branch
- You're more concerned with the final state than preserving commit history
- Time constraints make resolving individual conflicts impractical

### When Not to Use This Approach

Avoid this method when:
- Working on shared branches
- Commit history needs to be preserved
- Other developers might have based their work on your branch
- You need to maintain a clear audit trail of changes
