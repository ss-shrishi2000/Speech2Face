# Speech2Face

Aim of the project:
* Establishment of a strong connection between speech and appearance, part of which is a direct result of the mechanics of speech production: (age, gender, mouth shape, facial bone structure, thin or fuller lips).​ 
* Voice-Appearance correlations established from the way the person talks: language, accent, speed, pronunciations—such properties of speech are often shared among nationalities and cultures, which can in turn translate to common physical features.​ 
* Goal -> To design and develop a Deep-Learning Model that can be used to obtain inference of how a person looks from a short segment in which they shall be talking.​

#### Objectives:
* The objective of the project is to investigate how much information about a person's identity can be inferred directly from the way they speak.

#### Proposed Methodology:
​Our Speech2Face pipeline consists of two main components: ​

1) A voice encoder, which takes a complex spectrogram of speech as an input, and predicts a low-dimensional facial feature that would correspond to the associated face.​

2) A face decoder, which takes as input the facial feature and produces an image of the face in a canonical form (simplest form i.e. frontal-facing). During training, the face decoder is fixed, and we train only the voice encoder that predicts the facial feature. The voice encoder is a model we designed and trained.​

![voice2face](https://github.com/ss-shrishi2000/Speech2Face/assets/65821403/cdb808a1-9dad-4c37-b945-4ab8f41bf10d)

#### Generative Adversarial Networks:

* The basic idea behind speech-to-face conversion is to generate a realistic image of a person's face based on their spoken words. This requires the model to capture the complex relationships between speech and facial expressions, which can be difficult to do with traditional machine learning approaches.
* However, GANs are designed to learn complex, nonlinear relationships between input and output data, and can therefore be effective at generating realistic images of faces that match a given audio input.


![gan-123](https://github.com/ss-shrishi2000/Speech2Face/assets/65821403/41e33e3b-852f-4c63-a8c4-04bf2b2eac6a)

## FRAMES GENERATED FROM A 6 SEC VIDEO CLIP - TRAINING DATASET

![WhatsApp Image 2023-06-19 at 23 44 55 (1)](https://github.com/ss-shrishi2000/Speech2Face/assets/65821403/a93cfcfa-cb80-4943-973e-e427ccada80c)

![WhatsApp Image 2023-06-19 at 23 44 55](https://github.com/ss-shrishi2000/Speech2Face/assets/65821403/c83d338d-81e7-4a73-9412-48103d8d71e0)


#### GENERATED IMAGES FROM GAN MODEL OBTAINED AS FINAL OUTPUT:

![WhatsApp Image 2023-06-19 at 17 36 07](https://github.com/ss-shrishi2000/Speech2Face/assets/65821403/acd68f68-750e-4da8-9883-4e4de1847758)




### Results:

* Obtained up to 91 percent of accuracy in the results by inputting a short 6 sec Audio clip of a person talking into the system.
* Memory Utilisation: 28 GB
* RAM Utilisation: 12 GB
* For obtaining more clearer and visibly better images, we can opt for higher-end processing systems.

### Future Work:
* Utilising more higher end processing systems for obtaining more better picture quality along with clearer indications towards facial features.



