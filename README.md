# Contrastive Self-Supervised Classification of 3D Point Clouds using SimCLR and PointNet

In this project, we'll implement a 3D shape classification using a [PointNet](https://arxiv.org/abs/1612.00593) architecture, but we'll attempt to train the PointNet feature backbone using [SimCLR](https://arxiv.org/abs/2002.05709)&mdash;a contrastive self-supervised learning framework. We train our model on a synthetic dataset of 3D point cloud shapes from four primitive shapes: cubes, cylinders, cones, and tori; with random transformations to create a more diverse dataset.

# Experiments

We first verify our PointNet implementation by training a fully-supervised model on a relatively large dataset and manage to obtain an accuracy of 99.1%. We then add small transational noise to make training more challenging and our accuracy is now around ~75%


For SimCLR, we train our model on 6000 samples per shape without labels, and use biggest batch we can fit in our compute (in the range of 150 to 200) and train for 7-8 epochs&mdash;somewhat short of the 4096 batch size and 500+ epochs in the original paper. The classifier heads are then trained on labelled dataset 10\% of the size of unlabelled dataset. The results are shown in the table below.

<table>
  <tr>
    <td rowspan="2", colspan="1">Models</td>
    <td colspan="2">No translation noise</td>
    <td colspan="2">w/ translation noise</td>
  </tr>
  <tr>
    <td>Frozen</td>
    <td>Full</td>
    <td>Frozen</td>
    <td>Full</td>
  </tr>
  <tr>
    <td>SimCLR</td>
    <td>0.628</td>
    <td>0.980</td>
    <td>0.615</td>
    <td>0.721</td>
  </tr>
  <tr>
    <td>Vanilla</td>
    <td>0.436</td>
    <td>0.639</td>
    <td>0.375</td>
    <td>0.471</td>
  </tr>
</table>

---
More details are available in the report.
