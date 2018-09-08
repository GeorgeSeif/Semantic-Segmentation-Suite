Please fill out this issue template before submitting. **Issues which do not fill out this template, or are already answered in the FAQs will simply be closed.**

**Please go to Stack Overflow for help and support.** Also check past issues as many are repeats. Also check out the Frequently Asked Questions (FAQs) below in case your question has already been answered in an issue!

Issues should be one of the following:

1. Feature requests
2. Bug reports

------------------------

### Information
## Please specify the following information when submitting an issue:
- **What are your command line arguments?:**
- **Have you written any custom code?**:
- **What have you done to try and solve this issue?:**
- **TensorFlow version?**:

### Describe the problem
Describe the problem clearly here. Be sure to convey here why it's a bug or a feature request.

### Source code / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem.

### FAQs

- **Question:** I got an `InvalidArgumentError` saying that `Dimensions of inputs should match` **Answer:** See issue #17

- **Question:** Can you upload pre-trained weights for these networks? **Answer:** See issue #57

- **Question:** Do I need a GPU to train these models? **Answer:** Technically no, but I'd highly recommend it. I was able to train the models pretty well in about a day using a 1080Ti GPU. Training on CPU would take much longer than that.

- **Question:** Will you be adding the FCN or U-Net models? **Answer:** No I won't be adding those simply because they're a few years old and state-of-the-art has moved past that.

- **Question:** I got an invalid argument error when using the InceptionV4 model. Am I doing something wrong? **Answer:** No you're not! Due to the design of the InceptiveV4 model, when you end up upsampling you do some rounding which creates a shape mismatch. _This only happens when you end up having to use the `end_points['pool5']`_. See the code for some of the models if you want to check whether the model will use `end_points['pool5']`.
