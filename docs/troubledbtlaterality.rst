*****************************************************
Fixing accumulated AGD and laterality for Hologic DBT
*****************************************************

The code for extracting dose related information from Hologic digital breast tomosynthesis proprietary projection
images object used an incorrect tag to extract the laterality of the image in releases before 0.8.0 in June 2018. As a
result the accumulated AGD code didn't work, so the accumulated AGD cell on the mammography summary sheets remained
blank.

Releases between 0.8.0 and 0.10.0 had instructions on how to rectify this for existing studies in the database, but
these instructions are not suitable for version 1.0 and later and therefore these instructions have been removed.
