<img width="500" alt="Screenshot 2566-03-20 at 04 34 48" src="https://user-images.githubusercontent.com/30139280/226211093-2024ec2b-c20b-4f9a-8a6d-d677a93f4faa.png">

# SuperAI3-UWB-Classification Hackathon 
This is **House EXP's Super AI SS3 UWB Classification Code** that utilizes Pytorch, Timm, and HuggingFace!

<img width="500" alt="Screenshot 2566-03-20 at 04 41 08" src="https://user-images.githubusercontent.com/30139280/226211393-51c62971-3af0-40a6-8c7b-b3aae9225fc0.png">

The hackathon consisting of classifying signals from **Ultra Wide Band Sensors** readings of actions of people into the correct type. 

# Methodology
The key in here was that we decided to view the problem **not** only a **signal problem**, but as an **image classification** problem. That allowed us to turn these signals into things that were actually visualizable and easier to classify than real numbers.

## Data Processing
We experimented with a variety of data processing and transformations including: **wavelet transforms, fourier transforms, etc.** However it turned out that the best dataset we could get was using the teacher's original transformation code without any modifications.

<img width="500" alt="Screenshot 2566-03-20 at 04 46 37" src="https://user-images.githubusercontent.com/30139280/226211650-e2905cb6-50ec-44fc-96f0-1ef82a6e3336.png">


### Other Preprocesses
Click here for some of the transformations we tried:

<span>
<img width="500" alt="Screenshot 2566-03-20 at 04 49 02" src="https://user-images.githubusercontent.com/30139280/226211754-caea9c60-9609-4695-904f-a343b8608ab5.png">
<img width="500" alt="Screenshot 2566-03-20 at 04 49 13" src="https://user-images.githubusercontent.com/30139280/226211759-17ff326d-69b5-4fe4-b9aa-8a2389767511.png">
<img width="500" alt="Screenshot 2566-03-20 at 04 49 27" src="https://user-images.githubusercontent.com/30139280/226211774-a35b23e6-fe1d-4607-9c54-10f3e0fd8f3b.png">
<img width="500" alt="Screenshot 2566-03-20 at 04 49 46" src="https://user-images.githubusercontent.com/30139280/226211795-37bdcbab-6398-46be-8285-570901d531ca.png">
</span>

## Model Selection
After orignally trying **CNN** models which did not perform very well, I had the idea to try and switch to **Vision Transformers**. Which yielded very good results (boosting our score from 0.3 to 0.8). Therefore our team then explored other **SOTA** Vision Transformers model:

<span>

<img width="500" alt="Screenshot 2566-03-20 at 04 56 37" src="https://user-images.githubusercontent.com/30139280/226212072-fce5fca6-bf88-4b62-b857-4422b329a2c3.png">
<img width="500" alt="Screenshot 2566-03-20 at 04 36 15" src="https://user-images.githubusercontent.com/30139280/226211157-f7f3db82-65bd-4508-8d93-530546b4cf38.png">
<img width="500" alt="Screenshot 2566-03-20 at 04 57 26" src="https://user-images.githubusercontent.com/30139280/226212109-139066c0-8bbb-44db-a853-5db1a05ac5dd.png">
<img width="500" alt="Screenshot 2566-03-20 at 04 57 42" src="https://user-images.githubusercontent.com/30139280/226212121-f824fdbb-27a2-4439-a118-55e3dfc10406.png">

</span>

We ended up using the **MaxViT model** because of it's special properties:
- Multi Axis Attention
- Both block + grid attention
This meant that the model will be able capture the **sequentialness** of the data, leading to better perfomance than normal ViT Models. This hypothesis was validated in our testing.

Implementing the model, however, was also no easy task as it required me to write a seperate training loop to handle all the additional logic. However this custom training loop allowed us to incorporate more techniques than we have ever before.

## Other Techniques

<span>
<img width="500" alt="Screenshot 2566-03-20 at 05 03 23" src="https://user-images.githubusercontent.com/30139280/226212375-7d2dc4aa-1af2-402e-928d-1a0c4193e45e.png">
<img width="500" alt="Screenshot 2566-03-20 at 05 05 45" src="https://user-images.githubusercontent.com/30139280/226212451-93d55fd1-74d9-4d79-843b-14bf17b56aaa.png">
<img width="500" alt="Screenshot 2566-03-20 at 05 06 57" src="https://user-images.githubusercontent.com/30139280/226212512-4363d0ad-2221-4c17-b376-02d5d6f68291.png">
<img width="500" alt="Screenshot 2566-03-20 at 05 08 34" src="https://user-images.githubusercontent.com/30139280/226212588-a42f7f47-2609-4815-affe-8090701627ed.png">
</span>

In order to boost our score even futher we leveraged the following techniques:
- **Gradient Accumulation:** Simulating higher batch sizes without all the memory overhead, allowing the model to converge better.
- **Lookahead Optimization:** Using this relatively technique allowed us to keep two copies of the optimizer (fast and slow) and in theory would have made the model better.
- **Cross Validation:** This is crucial as our dataset was very small, and conducting cross-validation would mean that our model would be able to see all the data.
- **Pseudo Labeling:** Using the confident predictions from the model in order to predict the test data and reuse that as additional train data.
- **Weighted Ensembling:** We used this in order to combine the predictions from each model within the cross validation.
- **Voting Ensembling:** We then used this in order to combine the predictions from multiple different models.

# Credits:
- [Parin](https://github.com/parinzee): **ViT and Timm Training Scripts + Techniques**
- Film: **Initial Cross Validation Script + Experiments**
- Ten: **Model Tuning + Experiments**
- [Senmee](https://github.com/senmeetechin): **Data Preprocessing Scripts + Code Refactoring**
- Pond: **Data Preprocessing Scripts**
- Boss: **Data Preprocessing Scripts**
- Tae: **Data Preprocessing Scripts**
- Prae: **Hybrid Model Research + Experiments**
- Champ: **Consulting + Experiments**
- *EXP House:* **Relentless support, encouragement, and determination**

# Example Codes:
Our example codes can be accessed via jupyter notebook files in this repositories
- **SHARING_SignalPreparation.ipynb** for Data Preprocessing Script
- **SHARING_Timm_Training_For_UWB_Classification.ipynb** for TIMM Training and Interesting Techniques

# Final Results
<img width="800" alt="Screenshot 2566-03-20 at 04 36 38" src="https://user-images.githubusercontent.com/30139280/226211169-b4cf1409-3e5a-423c-9fbf-17b2f3c0287a.png">






