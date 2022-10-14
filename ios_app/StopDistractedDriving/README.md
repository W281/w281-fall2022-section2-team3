#  Distraction Detector

Classify driver images to identify distracted driving.

## Overview
This app uses a NN model to identify if driver in image is distracted. 

Each time a user selects a photo from the library or takes a photo with a camera,
the app passes it to a [Vision][Vision] image classification request.
Vision resizes and crops the photo to meet the model's constraints for its image input,
and then passes the photo to the model using the [Core ML][Core ML] framework behind the scenes.
Once the model generates a prediction, Vision relays it back to the app, which presents the results to the user.

This app has been built using [Lobe iOS Bootstrap] as reference.

[Vision]: https://developer.apple.com/documentation/vision

[Core ML]: https://developer.apple.com/documentation/coreml


[Lobe iOS Bootstrap]: https://github.com/lobe/iOS-bootstrap

[VNClassifyImageRequest]: https://developer.apple.com/documentation/vision/vnclassifyimagerequest

