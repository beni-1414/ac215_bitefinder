For a failure in text:
{
  "image_gcs_uri": "gs://bitefinder-data/user-input/chiggers_legimage111.jpg",
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

I cannot continue as vlmodel and rag need a docker-entrypoint similar to orchestrator to run as a service. Please take care of that.
