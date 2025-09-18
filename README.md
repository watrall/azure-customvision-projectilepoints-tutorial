# Projectile Point Classifier with Azure Custom Vision — From Labeled Photos to Deployed Classifier (Beginner Friendly)

## Table of Contents
- [Overview](#overview)
- [What You’ll Learn](#what-youll-learn)
- [What You’ll Need](#what-youll-need)
- [Understanding Computer Vision](#understanding-computer-vision)
- [Why Azure Custom Vision](#why-azure-custom-vision)
- [Understanding Bias in Machine Learning](#understanding-bias-in-machine-learning)
- [Ethics, Rights, Permissions, and Cultural Property](#ethics-rights-permissions-and-cultural-property)
- [Step-by-Step Tutorial](#step-by-step-tutorial)
  - [1) Create Azure resources](#1-create-azure-resources)
  - [2) Plan your dataset and taxonomy](#2-plan-your-dataset-and-taxonomy)
  - [3) Create an Azure Custom Vision project](#3-create-an-azure-custom-vision-project)
  - [4) Upload & label images](#4-upload--label-images)
  - [5) Train your first model in Azure Custom Vision](#5-train-your-first-model-in-azure-custom-vision)
  - [6) Evaluate like a researcher](#6-evaluate-like-a-researcher)
  - [7) Improve your dataset (iterate)](#7-improve-your-dataset-iterate)
  - [8) Publish the best iteration](#8-publish-the-best-iteration)
  - [9) Test the endpoint (portal-curl-python)](#9-test-the-endpoint-portal-curl-python)
  - [10) Responsible sharing](#10-responsible-sharing)
  - [11) (Optional) Build a no-code front end with Power Apps](#11-optional-build-a-no-code-front-end-with-power-apps)
- [Troubleshooting](#troubleshooting)
- [Contributing to this Tutorial](#contributing-to-this-tutorial)
- [License & Citation](#license--citation)
- [Glossary](#glossary)

---

## Overview

This tutorial will walk you through building a **machine learning classifier** for North American projectile points using **Azure Custom Vision**. By the end, you’ll have a **fully trained, deployed, and testable image classification model** that runs in the cloud. You’ll be able to upload a new image of a projectile point and receive a prediction of its type (for example, “Clovis” or “Dalton”), complete with a confidence score.

Why does this matter? Archaeologists, museum professionals, and students often face the challenge of sorting, labeling, and teaching typologies with large numbers of objects. A classifier doesn’t replace expertise, but it can act as a **research assistant**: helping you organize collections, test hypotheses, or give students hands-on experience with real datasets. Because everything is hosted on **Azure**, the system is accessible from anywhere, requires no local installs, and can even be integrated into teaching tools like Power Apps.

The larger goal here is trust. Many scholars are wary of AI because it feels opaque or overly technical. This tutorial shows that with **Azure Custom Vision**, you can build and evaluate models in a **transparent, step-by-step process**. You’ll see how the system learns, how it reports its strengths and weaknesses, and how you can refine it. By demystifying the process, the tutorial aims to help archaeologists and museum scholars use AI critically, responsibly, and confidently in their own work.


## What You’ll Learn

- Set up Azure resources on Free (F0) or Students tiers.  
- Create and label a dataset of projectile points.  
- Train, evaluate, and publish a classification model in Azure Custom Vision.  
- Call your model from the portal, cURL, and Python.  
- Build confidence in interpreting ML metrics like precision and recall.  
- Understand bias and limitations in AI systems.  
- (Optional) Create a no-code Power Apps front end for easy sharing.  
- Work ethically with heritage collections.  



## What You’ll Need

- A Microsoft account.  
- Access to Azure:  
  - [Azure Free Account](https://azure.microsoft.com/en-us/free/)  
  - [Azure for Students](https://azure.microsoft.com/en-us/free/students/) ($100 credit for 12 months, no credit card)  
  - [Microsoft Learn Training](https://learn.microsoft.com/en-us/training/)  
- A small set of rights-cleared projectile point images (start with ~50 images).  
  - **Image quality matters.** Clear, well-lit, in-focus images make it easier for the model to learn distinguishing patterns like fluting, base shape, or shoulder angles.  
  - Include variation: different lighting, backgrounds, and orientations. A model trained only on clean catalog shots may struggle with field photos.  
  - More images are always better—large, varied datasets usually produce more accurate models. But don’t worry if you’re starting small; even 50 images will let you complete this tutorial.  
- No installs required—everything runs in the browser.  


## Understanding Computer Vision

At its core, computer vision is about teaching a computer to recognize patterns in images. It doesn’t “understand” archaeology; it recognizes statistical fingerprints—edges, shapes, contours, colors, textures—that correlate with labels you provide. Show it dozens of labeled Clovis points and it starts to notice what they share, and how they differ from Dalton or Folsom.

A useful analogy: training a graduate student. At first, they can’t tell two types apart. After many labeled examples, they notice flute length, shoulder angle, base shape. They don’t have innate knowledge—they learn through exposure and feedback. A computer vision model is the same.

This distinction builds confidence. The system isn’t “deciding” what an object is—it’s calculating probabilities from patterns it has seen. “Clovis (0.85)” means the image fits the learned profile with 85% confidence. You still hold interpretive authority to accept, question, or refine the result.

Finally, performance depends on data. If your dataset varies in lighting, backgrounds, materials, and angles, the model becomes more robust. If it only includes pristine catalog shots, it may falter on field photos. Use AI critically, as a transparent assistant you can test, evaluate, and trust.



## Why Azure Custom Vision

When most people hear about AI, they think “black box.” That can feel intimidating, especially for scholars who work carefully with collections and archaeological data. The good news is Azure Custom Vision is designed to be transparent and beginner-friendly: you can build, test, and refine models entirely in your browser UI.

The key strength of Azure Custom Vision is that it lets you train AI on your own categories. Prebuilt services might recognize cars or coffee cups, but not “Clovis.” With Azure Custom Vision, you define your labels (Clovis, Dalton, Folsom, Kirk, etc.) and the model learns from your dataset—putting control in your hands as a scholar.

It also scales with your needs. Start small in a classroom, then grow into museum-scale datasets. It works on Free (F0) and Azure for Students tiers, so you can experiment without cost. And because uploading, labeling, training, and testing all happen in one place, it’s less overwhelming than advanced platforms.

Most importantly, Azure Custom Vision helps build trust. You get clear metrics—precision, recall, and a confusion matrix—to see where your model shines and where it struggles. This openness makes it easier to treat AI like a lab instrument you can calibrate and critique—not a magic oracle.


## Understanding Bias in Machine Learning

Bias in ML comes from two places:  
- Dataset bias: if you give the model many Dalton points but very few Clovis, it will learn Daltons better.  
- Model bias: Azure Custom Vision builds on base models trained on general imagery; that starting point shapes which patterns are easier to learn.  

The key is transparency: AI is not magic, but with honest evaluation, it can be a reliable assistant.  



## Ethics, Rights, Permissions, and Cultural Property

When working with museum collections and archaeological records, we need to remember that data is never “just data.” These are records of objects, histories, and sometimes living cultural knowledge.

- **Copyright and permissions:** Make sure you have the right to use the images you are training your model on.  
- **Cultural patrimony:** Many artifacts in museums are part of the cultural patrimony of descendant or source communities. Treat them with respect and consultation.  
- **Colonial histories:** Be transparent about the colonial contexts that shaped many collections.  
- **Privacy and sensitivity:** Be cautious about site locations, collector names, or donor histories.  
- **Bias and framing:** Typologies are scholarly constructs, not objective truths.  

This tutorial is designed as a teaching and research tool. It should be used responsibly, in collaboration with institutions and communities, and never as a substitute for expertise, consultation, or ethical stewardship.


# Step-by-Step Tutorial

## 1) Create Azure resources

Azure resources are the “infrastructure” your model needs: a container (Resource Group) and the Azure Custom Vision services that handle training and prediction. We’ll create two paired resources—Training and Prediction—and put them in the same region so publishing works smoothly.

**What:** You’re provisioning a Resource Group and paired Azure Custom Vision Training and Prediction resources in the same region.  
**Why:** Training builds the model; Prediction serves it via a web endpoint. Co-locating them ensures publishing works and reduces latency.  
**Validate:** In your Resource Group, you see two Custom Vision resources with identical Location values.

### Step-by-step
1. Open https://portal.azure.com and sign in.  
2. Click **Resource groups** → **+ Create**.  
   - Resource group name: `rg-customvision-points`  
   - Region: pick one close to you (e.g., East US).  
   - Click **Review + create** → **Create**.  
3. In the portal search bar, type **Custom Vision** → select it → click **+ Create**.  
   - Resource name: `cv-training-points`  
   - Type: Training  
   - Pricing tier: F0 (Free)  
   - Region: must match the Resource Group region  
   - Click **Review + create** → **Create**.  
4. Repeat step 3 to create a Prediction resource:  
   - Resource name: `cv-prediction-points`  
   - Type: Prediction  
   - Pricing tier: F0 (Free)  
   - Region: same as Training  
5. Open your Resource Group → confirm both appear and list the same Location.


## 2) Plan your dataset and taxonomy

Models only learn what you show them. Decide your categories (labels), aim for balance, and gather images that vary in lighting, background, angle, and material. Hold back ~10–20% as a manual test set you won’t upload.

**What:** You’re creating a labeled, balanced set of images that reflects real-world variation, plus a small held-out test set.  
**Why:** Balanced variety improves generalization; a held-out set keeps evaluation honest.  
**Validate:** Counts per label are roughly equal, and you have a separate manual-test folder.

### Step-by-step
1. Choose 3–5 types (e.g., Clovis, Folsom, Dalton, Kirk).  
2. Target 30–50 images per type to start.  
3. On your computer, create a folder structure where each projectile point type has its own folder. For example: place all your Clovis images into a folder named *clovis*, all your Folsom images into *folsom*, and so on. The folder names should match the categories you plan to use in Azure Custom Vision.  
4. Record a mapping table (CSV/Markdown) of filename → label.  
5. Move 10–20% per class into a separate folder to act as your held-out test set (do not upload these).


## 3) Create an Azure Custom Vision project

Now that your resources exist, you need a project to actually hold your dataset and training runs. Think of this like opening a fresh lab notebook: everything for this classifier will live inside it.

**What:** You’re creating a new Azure Custom Vision project that links to your Training and Prediction resources.  
**Why:** The project is the workspace where you upload images, apply labels, train models, and compare iterations.  
**Validate:** The project appears in the Custom Vision portal with the classification type you selected.

### Step-by-step
1. Go to https://customvision.ai and sign in.  
2. Click **New Project**.  
   - Name: `ProjectilePoints`  
   - Description: `Classifier for North American projectile point types`  
   - Resource Group: select `rg-customvision-points`  
   - Training resource: choose `cv-training-points`  
   - Prediction resource: choose `cv-prediction-points`  
   - Project type: **Classification**  
   - Classification type: **Multiclass (single tag per image)**  
   - Domain: **General**  
3. Click **Create Project**.  
4. The project dashboard opens — this is where you’ll upload images and train models.


## 4) Upload & label images

With your project created, it’s time to give the model something to learn from. Each image needs to be tagged so the system knows what it represents.

**What:** You’re uploading projectile point images and tagging them with their type.  
**Why:** Tags are the “answers” the model uses during training. Accurate labeling is critical for performance.  
**Validate:** Each uploaded image has a tag visible under its thumbnail in the portal.

### Step-by-step
1. In your project dashboard, click **Add images**.  
2. Select one of your folders (e.g., *clovis*) and upload all the images inside.  
3. When prompted, type the tag name (e.g., `Clovis`) and click **Enter**.  
4. Repeat the upload/tagging process for each type (Folsom, Dalton, Kirk, etc.).  
5. Verify: In the left-hand tag list, you should see each type you created. Click a tag to filter and confirm the images are correctly assigned.  
6. If you spot mistakes, click an image, edit its tags, and save.  


## 5) Train your first model in Azure Custom Vision

Now you let Azure do the heavy lifting: turning labeled images into a working model.

**What:** You’re training an initial model on your labeled dataset.  
**Why:** Training identifies visual patterns that separate one class from another. This first run gives you a baseline.  
**Validate:** A new iteration appears in the portal with performance metrics.

### Step-by-step
1. In your project dashboard, click **Train**.  
2. Choose **Quick Training** (fast, good for small datasets).  
3. Click **Train**.  
4. Wait — this can take 1–10 minutes depending on dataset size.  
5. When training finishes, you’ll see **Iteration 1** with metrics: precision, recall, and average precision.  


## 6) Evaluate like a researcher

Numbers matter — but only if you know how to read them. Azure gives you precision, recall, and a confusion matrix.

**What:** You’re interpreting model metrics to judge reliability.  
**Why:** Evaluation tells you whether the model is trustworthy and where it struggles.  
**Validate:** You can identify which classes are strong and which need more examples.

### Step-by-step
1. After training, review the **Performance** tab.  
2. **Precision**: of the images predicted as Clovis, what percentage were truly Clovis?  
3. **Recall**: of all Clovis images in the dataset, what percentage did the model correctly find?  
4. **Confusion Matrix**: shows where misclassifications happen (e.g., Dalton images mislabeled as Kirk).  
5. Compare metrics across tags. If one type lags, plan to add more or better images for it.  


## 7) Improve your dataset (iterate)

Machine learning is not one-and-done. You refine, retrain, and improve.

**What:** You’re expanding or cleaning your dataset, then retraining.  
**Why:** More examples and better labels usually boost performance and robustness.  
**Validate:** Iteration 2 (and beyond) show higher or more balanced metrics.

### Step-by-step
1. Identify weak classes from Step 6.  
2. Collect or photograph more examples for those types.  
3. Upload them into your project with correct tags.  
4. Retrain: click **Train** again → choose **Quick Training**.  
5. A new iteration appears (Iteration 2).  
6. Compare Iteration 1 vs Iteration 2. Which metrics improved?  


## 8) Publish the best iteration

Once you’re satisfied, publish so the model is available via a prediction endpoint.

**What:** You’re publishing your trained iteration to your Prediction resource.  
**Why:** Publishing creates a web-accessible endpoint for real-world testing.  
**Validate:** The iteration has a green “Published” label with an endpoint URL.

### Step-by-step
1. In the **Performance** tab, find the iteration you want.  
2. Click **Publish**.  
3. Enter a name, e.g., `projectilepoints-classifier`.  
4. Select your Prediction resource (`cv-prediction-points`).  
5. Click **Publish**.  
6. Go to the **Prediction URL** tab — copy the endpoint and prediction key. You’ll need them for cURL/Python.  


## 9) Test the endpoint (portal, cURL, Python)

Time to check your deployed model. You’ll test in three ways: in the portal, from the terminal (cURL), and in Python.

**What:** You’re sending an image to your model and receiving a prediction.  
**Why:** Testing ensures your endpoint works and you can integrate it into workflows.  
**Validate:** The response shows the correct type with a confidence score.

### Step-by-step (Portal)
1. In your project, click **Quick Test**.  
2. Upload an image from your held-out test set.  
3. See the predicted tag and probability.  

### Step-by-step (cURL)
1. On your computer, open a terminal:  
   - **Windows:** Press **Start**, type `cmd`, hit Enter.  
   - **macOS/Linux:** Open **Terminal** from Applications/Utilities.  
2. Run this command (replace `PREDICTION_URL` and `PREDICTION_KEY`):  

```bash
curl -X POST "PREDICTION_URL"   -H "Prediction-Key: PREDICTION_KEY"   -H "Content-Type: application/octet-stream"   --data-binary @testimage.jpg
```

3. The response JSON shows tags and probabilities.  

### Step-by-step (Python)
1. Save an image (e.g., `testimage.jpg`) in your working directory.  
2. Use the provided script:  

```bash
python scripts/predict.py testimage.jpg
```

Make sure you set environment variables first:  

- macOS/Linux:  
  ```bash
  export PREDICTION_URL="https://<region>.api.cognitive.microsoft.com/customvision/v3.0/Prediction/<projectId>/classify/iterations/<publishedName>/image"
  export PREDICTION_KEY="<your-prediction-key>"
  ```

- Windows PowerShell:  
  ```powershell
  $env:PREDICTION_URL="https://<region>.api.cognitive.microsoft.com/customvision/v3.0/Prediction/<projectId>/classify/iterations/<publishedName>/image"
  $env:PREDICTION_KEY="<your-prediction-key>"
  ```


## 10) Responsible sharing

Now that your model works, think carefully about how it’s shared and used.

**What:** You’re documenting limitations, rights, and appropriate contexts for use.  
**Why:** Transparency prevents misuse and builds scholarly trust.  
**Validate:** Your README or project notes explain scope and cautions.

### Step-by-step
1. In your project documentation, describe:  
   - Dataset size and source.  
   - Classes included.  
   - Known limitations or biases.  
2. State clearly: “This classifier is a teaching and research tool. It does not replace expert analysis.”  
3. Share responsibly: avoid uploading sensitive images or site coordinates.  


## 11) (Optional) Build a no-code front end with Power Apps

For colleagues or students without coding skills, you can make a drag-and-drop interface.

**What:** You’re connecting your Azure Custom Vision endpoint to a Power App.  
**Why:** A simple web/mobile app makes the classifier accessible in classrooms or exhibits.  
**Validate:** A user can upload an image in the app and see the model’s prediction.

### Step-by-step
1. Go to https://make.powerapps.com.  
2. Create a new **Canvas App**.  
3. Add a button: “Upload image.”  
4. Connect to **Custom Vision** via the AI Builder connector.  
5. Paste your Prediction URL and key.  
6. Add a text box to display the returned tag and probability.  
7. Save and share the app with colleagues.  


# Troubleshooting

- **Publish fails?** Training and Prediction resources must be in the same region.  
- **Unauthorized?** Use the correct Prediction Key, not the Training Key.  
- **Class imbalance?** Add more images to underrepresented classes.  
- **Overfitting?** Remove near-duplicate images or add more varied examples.  



# Contributing to this Tutorial

We welcome improvements. If you want to tweak the tutorial, fix bugs, or add examples:

1. Fork this repository.  
2. Clone your fork (or open in GitHub Codespaces).  
3. Make your changes: README, examples, code.  
4. Commit and push to your fork.  
5. Open a Pull Request back to this repo — explain what changed and why.  



# Citation

If you use or adapt this tutorial, please cite the repository. Include the provided `CITATION.cff` so GitHub can generate APA/BibTeX automatically (look for “Cite this repository” on the repo page).  



# Glossary

- **Class/Tag:** The label applied to each image (e.g., Clovis).  
- **Precision:** Of images predicted as Clovis, what % were correct.  
- **Recall:** Of all Clovis images, what % were found.  
- **Confusion Matrix:** A table showing misclassifications.  
- **Overfitting:** When a model memorizes training data but fails on new images.  
- **Endpoint:** The web address you send images to for predictions.  
- **Prediction Key:** A secret key used to authenticate endpoint requests.  
