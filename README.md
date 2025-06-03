# lovegpt
This is a proof of concept of a new kind of therapy / coaching app that utilizes AI to analyze user communications with their partner or family member, and determines what is causing the conflict or misunderstanding. It provides numeric scores, and color coded commentary on each message between the parties, and suggests alternative phrasing or compromise proposals that would be better ways to handle the issue of concern.

While the app has not been maintained for some time, it is fully functional and actually can be quite useful - the creator developed it 2 years ago when he and his partner were going thru a rough patch and had become separated, and was able to patch things up by following the advice given by this app after pasting in the transcripts of some arguments they'd had over whatsapp. Apparently it was good advice, because they got married last week!

## Installation and Use

`
cd server
OPENAI_API_KEY=<your api key> && npm run start
`

## Change LLM or API Vendor
This app works with any model capable of following instructions re: output formatting and served via a chat completions compatible API. By default, it uses the OpenAI API and GPT-4.1 as the model, but you can easily switch to any other high end model by setting a custom API endpoint to point to Openrouter and changing the model ID as desired.

You can set a custom API endpoint and/or change the model by setting the OPENAI_API_ENDPOINT and LOVEGPT_MODEL environment variables accordingly. For example, here's how you would run LoveGPT with a Gemini 2.5 Pro backend via Openrouter:

`
OPENAI_API_KEY=<your openrouter api key> OPENAI_API_ENDPOINT=https://openrouter.ai/api/v1 LOVEGPT_MODEL=google/gemini-2.5-pro-preview npm run start
`

## Known Issues

This is just a proof of concept / prototype and was developed in mid-2023, prior to the introduction of models that return JSON reliably. Instead, we obtain a JSON response using few-shot in context learning (training examples provided as a prequel to the conversational request), which worked reliably with the original gpt-4 model that powered this prototype when it was developed. It will also work with any of the newer models released since then, and no code changes are required to run this with cutting edge mid-2025 technology. However, if you want to contribute to this project we highly recommend switching to a json_object response format!

The UI is also quite basic and built using jQuery... It is recommended that it be redone in React or another framework that is scalable and modular.

Finally, to make this into a viable mainstream product, there need to be more ways of providing the model with the data it needs to analyze - pasting in whatsapp transcripts only addresses a limited slice of possible use cases. We suggest putting this in some kind of hybrid mobile app framework or just sticking it directly into an Android or iOS WebView / WKWebView container, and then registering the app as a valid share target for all text-based data and hooking that into the webapp which can start when such a share is received. This will let users easily select and share mesages or emails from any other app on their phone or from a browser. 
