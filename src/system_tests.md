## Launch the /docs visualization in a browser and try the following test cases:

### orchestrator input evaluation:

For a failure in text:
{
  "image_gcs_uri": "gs://bitefinder-data/images/testing/chiggers/chiggers_legimage111.jpg",
  "user_text": "I was bitten while I was sleeping I believe",
  "overwrite_validation": false,
  "first_call": true,
  "history": [
  ],
  "return_combined_text": true,
  "debug": false
}


For a failure in image:
{
  "image_gcs_uri": "gs://bitefinder-data/data/images/skin_of_color_testing/bed_bugs/bed_bug_d_1.jpg",
  "user_text": "I was bitten while I was sleeping I believe, the bite is itchy.",
  "overwrite_validation": false,
  "first_call": true,
  "history": [
  ],
  "return_combined_text": true,
  "debug": false
}

For all good:
{
  "image_gcs_uri": "gs://bitefinder-data/user-input/chiggers_legimage111.jpg",
  "user_text": "I was bitten while I was sleeping at home I believe, the bite is itchy.",
  "overwrite_validation": false,
  "first_call": true,
  "history": [
  ],
  "return_combined_text": true,
  "debug": false
}

Should return a prediction of chiggers based on our vlmodel.

### Rag endpoint:

{
  "question": "How can I treat this bite?",
  "symptoms": "itchy red bite",
  "conf": 0.6,
  "bug_class": "tick",
  "session_id": "",
}


{
  "question": "Do you know places in the US i could go buy rubbing alcohol? I am not from this country",
  "symptoms": "itchy red bite",
  "conf": 0.6,
  "bug_class": "tick",
  "session_id": "GET THE PREVIOUS SESSION ID FROM THE ORCHESTRATOR OUTPUT",
}
