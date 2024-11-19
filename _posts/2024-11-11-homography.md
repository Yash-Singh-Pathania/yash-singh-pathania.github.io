---
title: "Homography For Old Art Pictures "
date: 2024-09-11
permalink: /posts/on_my_campus
categories: [UCD, Technology, Student Life]
tags:
  - python
  - fastApi
  - SelfProject
  - Streamlit
  - PostgreSQL
  - Render
mermaid: true
author: Yash Singh Pathania
---

# Unraveling the Magic of Homography Transformations: A Coding Adventure

Welcome to a fascinating exploration of homography transformations—a powerful technique in computer vision that adjusts the perspective of an image. This process has a myriad of applications, from correcting distorted photos to automating image preprocessing for machine learning tasks. In this post, I'll walk you through how we built a simple yet effective Homography Transformation App using React for the frontend and FastAPI for the backend.

### What is Homography?

Homography is a transformation applied to points on a plane in a three-dimensional projective space. It's commonly used to manipulate images where you need to correct the perspective. For example, imagine taking a picture of a painting from the side; homography can help "straighten" that image to look as if you'd captured it head-on.

### The Mathematical Grit

Homography involves a series of linear transformations and is defined by a 3x3 matrix. Given a set of four points in both the original and target images, we can compute this matrix to map between these points. Here’s a simplified breakdown of the mathematics involved:

1. **Point Collection**:
   - You define points on your image that form a quadrilateral, which might represent, for instance, the corners of a document in an image taken at an angle.

2. **Matrix Equation**:
   - The homography matrix \( H \) relates corresponding points in two images, \( p \) in the original image and \( p' \) in the transformed image, as follows:
   \[
   p' = H \cdot p
   \]
   Where \( p \) and \( p' \) are points in homogeneous coordinates.

3. **Direct Linear Transformation (DLT)**:
   - The matrix \( H \) can be found using a series of linear equations derived from the coordinates of points in both images:
   \[
   \begin{align*}
   ax + by + c & = x' \\
   dx + ey + f & = y' \\
   gx + hy + i & = 1
   \end{align*}
   \]
   - The coefficients (a-i) are solved using Singular Value Decomposition (SVD), ensuring the matrix transforms the source points to their destinations.

4. **Matrix Application**:
   - For any given point \( (x, y) \) in the original image, we apply \( H \) to find the corresponding point in the new image, effectively transforming the perspective.

### Building the App

#### Step 1: Setting Up the Environment

We started by setting up two main components: a React frontend and a FastAPI backend. React manages the UI, providing a responsive way to upload images and select points. FastAPI, on the other hand, handles the computationally intensive task of calculating and applying the homography matrix.

#### Step 2: User Interaction

Users upload an image and then click to select four points on the image. These points dictate how the image will be transformed. The app dynamically draws a quadrilateral as points are selected, reinforcing the expected transformation.

#### Step 3: Backend Magic

Upon receiving the points and the image, the backend:
- Orders the points to ensure consistency.
- Computes the homography matrix using the provided points.
- Applies this matrix to transform the image, which is then sent back to the user.

### Challenges and Solutions

Implementing the app wasn't without its hurdles. Ensuring that points are selected in a way that makes mathematical sense (i.e., forming a convex quadrilateral) required some additional checks. Moreover, handling different image sizes and aspect ratios meant scaling the points appropriately before applying transformations.

### Conclusion: Why It Matters

The Homography Transformation App is more than just a technical showcase; it’s a practical tool for anyone needing to correct image perspectives quickly. For developers, it’s a great starting point for more complex computer vision tasks that require image normalization.

This project is a testament to the beauty of combining mathematical theory with software development to solve real-world problems. Whether you’re a seasoned developer or a curious enthusiast, the world of homography transformations offers endless possibilities for exploration and innovation.

---