
# Step-by-Step Guide: Using Git LFS to Push Large Files to GitHub

If your file exceeds GitHub's 100 MB file size limit, use Git Large File Storage (Git LFS) to manage and push it. Follow these steps:

---

## 1. Install Git LFS

### **On macOS**
Install Git LFS using Homebrew:
```bash
brew install git-lfs
```

### **On Windows**
1. Download and install Git LFS from [git-lfs.github.com](https://git-lfs.github.com).
2. Verify the installation by opening a terminal (Command Prompt, PowerShell, or Git Bash) and running:
   ```bash
   git lfs --version
   ```

---

## 2. Initialize Git LFS in Your Repository

1. Navigate to your project folder in the terminal:
   ```bash
   cd path\to\your\project
   ```

2. Enable Git LFS for your repository:
   ```bash
   git lfs install
   ```

---

## 3. Track Your Large File

1. Identify the large file(s) in your project. For example: `models/resNet50_model.pth`.

2. Track the file with Git LFS:
   ```bash
   git lfs track "models/resNet50_model.pth"
   ```

   This command updates (or creates) a `.gitattributes` file to include the tracked file(s).

---

## 4. Stage and Commit Changes

1. Add the `.gitattributes` file and the large file(s) to Git:
   ```bash
   git add .gitattributes
   git add models/resNet50_model.pth
   ```

2. Commit the changes:
   ```bash
   git commit -m "Track resNet50_model.pth using Git LFS"
   ```

---

## 5. Push to GitHub

Push your changes to GitHub, setting the upstream branch if needed:
```bash
git push --set-upstream origin master
```

Git LFS will upload the large file(s) separately.

---

## 6. Verify the Upload

1. Go to your GitHub repository in your browser.
2. Check that the large file appears as a pointer file. It should look like this:
   ```
   version https://git-lfs.github.com/spec/v1
   oid sha256:<file hash>
   size <file size>
   ```

---

## Troubleshooting Common Issues

### **Error: File Exceeds GitHub's Size Limit**
- Ensure the file is being tracked by Git LFS. Check tracked files:
  ```bash
  git lfs track
  ```

### **Error: Failed to Push Large File**
- If the file was committed without LFS, remove it from Git history and recommit using LFS:
  ```bash
  git rm --cached models/resNet50_model.pth # remove **model.pth from the models folder
  git rm --cached *.md ## remove all files with the extension of *.md
  git commit -m "Remove large file from history"
  git push origin master
  ```

### **Reinitialize Git LFS**
- If LFS tracking is not working as expected, reinitialize it:
  ```bash
  git lfs uninstall
  git lfs install
  ```

---

## Notes for macOS and Windows Users

- **GitHub LFS Free Limit**: GitHub provides 2 GB of free LFS storage for public repositories. For private repositories or larger limits, consider upgrading to a paid plan.
- **File Compression**: To stay under limits, compress large files before committing.

---

Now you’re all set to manage large files on GitHub using Git LFS. If you encounter any issues, revisit the steps above or consult the [Git LFS documentation](https://git-lfs.github.com).
