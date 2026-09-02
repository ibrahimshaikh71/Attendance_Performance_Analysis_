# GitHub Authentication Guide for Attendance Project

## Quick Setup (Recommended)

### Step 1: Create Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Tokens (classic)"**
3. Fill in:
   - **Token name**: `Attendance-Project-Token`
   - **Expiration**: 90 days
   - **Scopes**: Select ONLY these:
     - ✓ `repo` (Full control of private repositories)
     - ✓ `workflow` (Update GitHub Action workflows)
4. Click **"Generate token"**
5. **COPY the token** (you won't see it again)

### Step 2: Push to GitHub
Run this in PowerShell:

```powershell
cd "c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project"

# Set up credential storage
git config --global credential.helper wincred

# Push to GitHub
git push origin main
```

**When prompted:**
- **Username**: `ibrahimshaikh71`
- **Password**: Paste your token from Step 1

---

## Alternative: SSH Setup

If you prefer SSH (more secure, no token needed):

### Step 1: Generate SSH Key
```powershell
ssh-keygen -t ed25519 -C "shaikhibrahim1204@gmail.com"
```
Press **Enter** 3 times (no passphrase needed)

### Step 2: Add Public Key to GitHub
1. Copy your public key:
```powershell
cat ~/.ssh/id_ed25519.pub
```

2. Go to: https://github.com/settings/keys
3. Click **"New SSH key"**
4. Paste the key and name it `My Windows Machine`

### Step 3: Configure Git for SSH
```powershell
cd "c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project"
git remote set-url origin git@github.com:ibrahimshaikh71/Attendance_Performance_Analysis_Project.git
git push origin main
```

---

## Verify Success

After pushing, check:

1. **Terminal output** should show:
   ```
   To https://github.com/ibrahimshaikh71/Attendance_Performance_Analysis_Project.git
    * [new branch]      main -> main
   ```

2. **Visit your repository**:
   https://github.com/ibrahimshaikh71/Attendance_Performance_Analysis_Project

3. **Verify files uploaded** - You should see all your Python files, CSV, README, etc.

---

## Troubleshooting

### "404 Not Found"
- Repository doesn't exist yet
- **Solution**: Create it first at https://github.com/new

### "Authentication Failed"
- Wrong token or username
- **Solution**: 
  ```powershell
  git credential-wincred erase host=github.com
  git push origin main
  ```
  Then enter correct credentials

### "Permission Denied (publickey)"
- SSH key not added to GitHub
- **Solution**: Go to https://github.com/settings/keys and add your public key

---

## Your Project Details

- **Local Path**: `c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project`
- **GitHub URL**: https://github.com/ibrahimshaikh71/Attendance_Performance_Analysis_Project
- **Username**: `ibrahimshaikh71`
- **Email**: `shaikhibrahim1204@gmail.com`
- **Files Ready**: 18 files committed and ready to push

