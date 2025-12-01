This repository contains the backend for Team TARBall's submission to the G7 GovAI Grand Challenge.

After running main, the sample proposal text can be submitted with the following command:

curl -F "file=@proposal_text.txt" http://localhost:8002/submit/

This will return an ID number, which is used in the following command to return the system's response: 

curl  http://localhost:8002/get-response/[#####]