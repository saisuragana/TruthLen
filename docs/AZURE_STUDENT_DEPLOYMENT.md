# Deploy TruthLen with Azure for Students

This guide deploys the existing FastAPI app and dashboard to Azure Container Apps. It uses the student subscription and a scale-to-zero container so the demo is not left running all day. A public HTTPS URL is generated for the organizers.

## Before starting

- Activate Azure for Students using the eligible student's own college email. Microsoft requires the account holder to be at least 18, enrolled full time at an accredited degree-granting college or university, and verified through the institution. The offer does not require a credit card and provides $100 of credit for 12 months. If the credit runs out, the subscription is disabled unless the account holder chooses to upgrade.
- Use the student's Azure account; do not share passwords. The subscription owner should do the deployment and keep the resulting URL.
- Azure Container Apps can scale to zero. The first visit after inactivity may take longer while the app starts. The first model check can also take several minutes because model weights download at runtime.
- This demo uses local container storage. Analysis history and downloaded model weights may disappear after the app scales to zero or is replaced. The uploaded media is deleted after analysis by default.
- Do not add `.env`, `eval/`, uploaded photos, model weights, or API keys to the deployment. `.dockerignore` excludes them.

## 1. Install Azure CLI on Windows

Open **PowerShell** and run:

```powershell
winget install --exact --id Microsoft.AzureCLI
```

When it finishes, close PowerShell and open a new PowerShell window so the `az` command is available.

## 2. Sign in and confirm the student subscription

```powershell
az login
az account list --output table
```

Find the subscription named **Azure for Students**. If it is not the selected one, copy its subscription ID from the list and run:

```powershell
az account set --subscription "PASTE-STUDENT-SUBSCRIPTION-ID"
az account show --output table
```

Do not continue if the selected subscription is not the student subscription.

## 3. Open the TruthLen folder

```powershell
Set-Location "C:\Users\dattu\Documents\Codex\2026-11-02\referenced-chatgpt-conversation-this-is-an\TruthLen"
```

Check that the root `Dockerfile` is present:

```powershell
Get-ChildItem Dockerfile
```

## 4. Deploy the public app

This creates a resource group and Container App in Central India, builds from the root Dockerfile, and prints the public URL:

```powershell
az containerapp up `
  --name truthlen-demo `
  --resource-group truthlen-demo-rg `
  --location centralindia `
  --source . `
  --ingress external `
  --target-port 8000
```

The first build can take several minutes. If Azure asks to register a provider or install the Container Apps CLI extension, answer **Y**. Keep the PowerShell window open until deployment completes and copy the HTTPS URL printed at the end.

## 5. Limit replicas and set enough memory for inference

After deployment, run:

```powershell
az containerapp update `
  --name truthlen-demo `
  --resource-group truthlen-demo-rg `
  --cpu 4 `
  --memory 8Gi `
  --min-replicas 0 `
  --max-replicas 1
```

This limits the app to one CPU-only replica and lets it shut down while idle. Container Apps Consumption has a monthly free allowance, but this larger model container may go beyond that allowance. The student credit covers eligible Azure usage; watch **Cost Management** in the Azure portal. The subscription will be disabled if student credit is exhausted unless its owner upgrades it.

## 6. Check the site

Replace the example host with the URL Azure printed:

```powershell
Invoke-RestMethod "https://YOUR-APP-HOST/api/health"
```

Then open `https://YOUR-APP-HOST` in a browser. Test one small image, then an audio or short video. The app may download model weights on the first analysis, so wait for the result. Keep the caveat visible to organizers: model scores are fallible screening signals, not proof of whether media is real or AI-generated.

## If Azure reports a quota or region error

Try `eastus` in place of `centralindia` in the deploy command. Do not create a paid plan or upgrade the subscription. If the student subscription does not allow Container Apps or the deployment fails with a quota message, copy the exact error text and use the Azure portal's **Cost Management** page before trying another resource.

## After the event

When you no longer need the public demo, remove its Azure resources from PowerShell:

```powershell
az group delete --name truthlen-demo-rg --yes --no-wait
```

This removes the demo app and its associated resources. Do not run this until the organizers have finished checking the link.
