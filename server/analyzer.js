const { Configuration, OpenAIApi } = require('openai');


/* Get config from environment variables or use reasonable defaults */
const API_KEY = process.env.OPENAI_API_KEY || ''
const API_ENDPOINT= process.env.OPENAI_API_ENDPOINT || "https://api.openai.com/v1/"
const MODEL = process.env.LOVEGPT_MODEL || "gpt-4.1"

const configuration = new Configuration({
    apiKey: API_KEY,
    baseURL: API_ENDPOINT 
});

//requests a big-picture summary based on the results of the message-by-message analysis
//the goal is to describe what the conversation was about, the ways in which it was healthy or unhealthy, and to give both partners
//feedback on their own role in the discussion and (if needed) actionable advice for each person on how they can improve their relationship
//we do not recommend running this prompt directly against raw conversation data; instead, first run the analysis prompt
//so that when we request summarization, the context already contains the training examples, the raw conversation, and the message-by-message analysis
//
//recommend model: gpt-3.5-16k... if you use GPT4, disregard the above advice, and request summarization at zero shot, which will work fine, and save you some money
const summarizationPrompt = `
Thank you. Now please provide your overall big picture analysis, advice for the user, and advice for the partner, in the following format:

{
    "overall_analysis": "...",
    "advice_for_user": "...",
    "advice_for_partner": "...",
}
`

//request that a conversation be analyzed one message at a time
const analysisPrompt = `
Please analyze the following conversation between a user and their partner, and give each message a health score of "positive", "negative", or "neutral" based on whether it is good for the relationship, bad for the relationship, or neither. Please also provide a description of the emotional tone of the message in 5 words or less (e.g. "Happy", "Anxious", "Curious and inquisitive"), and include a running interaction health score. This should be a floating point number between 1.00 and 10.00, and reflect the *overall* health of the *interaction* up to that point in time (not the quality of the specific message). Each message should also include your commentary regarding that message.

    Please provide your analysis as a JSON array where each item represents a message in the conversation. Here is an example of the format to use:
    
    [{
      "turn": "user", //or "partner" 
      "message_text": "...",
      "health_score":  "positive",
      "emotional_tone_description": "Positive and caring"
      "interaction_health_score":  6.38,
      "commentary":  "The user is offering to pick up dinner and communicates clearly about when they will be home, despite the partner's evasive response when asked about the friend they're hanging out with."
    }, { ... }]
    
Conversation:
`

//the initial fine tuning instruction, specifying the high level purpose of the request and the desired output format
const systemInstruction = `You are a specialist in analyzing interactions between partners. Your output should always be valid JSON without additional content, so that it can be parsed by a javascript client`

//An example of the kind of raw data the model will be receiving from the user
const trainingConversation = `Hey honey, how was your day?
Partner
Hey. It was alright, I guess. Hung out with some people from work. You?
User
I'm still at the office... they gave me a 5k imac finally so I'm catching up on color grading. What are you up to right now?
Partner
Oh, just hanging out at home... with a friend. That's cool about the iMac though. Are you going to be late again?
User
Who's the friend?
Partner
Just someone I met recently, Chris. We've been hanging out a bit lately. Don't worry about it, okay? It's just casual. How much longer do you think you'll be at the office?
`

//An example of the desired output given the example conversation data.
//If using GPT4, this is not required, the single example in the prompt is sufficient. But if using 3.5-turbo then it needs the extra context
const trainingAnalysis = `
    [
        {
        "turn": "user",
        "message_text": "Hey honey, how was your day?",
        "health_score": "positive",
        "emotional_tone_description": "Positive and caring",
        "interaction_health_score": 8.3,
        "commentary": "The user initiates the conversation with a warm greeting and expresses genuine interest in the partner's day."
        },
        {
        "turn": "ChatGPT",
        "message_text": "Hey. It was alright, I guess. Hung out with some people from work. You?",
        "health_score": "neutral",
        "emotional_tone_description": "Neutral with a hint of reservation",
        "interaction_health_score": 6.9,
        "commentary": "The partner responds but seems slightly reserved about the details of their day. The tone is not negative but not very enthusiastic either."
        },
        {
        "turn": "user",
        "message_text": "I'm still at the office... they gave me a 5k imac finally so I'm catching up on color grading. What are you up to right now?",
        "health_score": "positive",
        "emotional_tone_description": "Informative and mildly excited",
        "interaction_health_score": 7.1,
        "commentary": "The user provides information about their current activity and exhibits mild excitement about the new equipment. The user also inquires about the partner's current activity."
        },
        {
        "turn": "partner",
        "message_text": "Oh, just hanging out at home... with a friend. That's cool about the iMac though. Are you going to be late again?",
        "health_score": "negative",
        "emotional_tone_description": "Evasive and slightly concerned",
        "interaction_health_score": 5.3,
        "commentary": "The partner provides a vague answer and seems a bit concerned about the user's work schedule."
        },
        {
        "turn": "user",
        "message_text": "Who's the friend?",
        "health_score": "positive",
        "emotional_tone_description": "Curious and inquisitive",
        "interaction_health_score": 6.3,
        "commentary": "The user becomes curious about the unidentified friend. This demonstrates a natural concern and interest in the partner's activities."
        },
        {
        "turn": "ChatGPT",
        "message_text": "Just someone I met recently, Chris. We've been hanging out a bit lately. Don't worry about it, okay? It's just casual. How much longer do you think you'll be at the office?",
        "health_score": "negative",
        "emotional_tone_description": "Defensive and redirecting",
        "interaction_health_score": 5.0,
        "commentary": "The partner seems defensive about their relationship with Chris and tries to redirect the conversation back to the user's schedule."
        }
    ]          
`
const openai = new OpenAIApi(configuration);
const trainingData = [
    { role: 'system', content: systemInstruction },
    
    { role: 'user', content: `${analysisPrompt}
                              ${trainingConversation}`}, 
    {role: "assistant", "content": trainingAnalysis}];


async function askQuestion(detailedAnalysis, question) {
    const questionTemplate = `
    Question: ${question}
    Context:
    
    ${JSON.stringify(detailedAnalysis)}`
    console.log(questionTemplate)
    const response = await openai.createChatCompletion({
        model: 'gpt-4',
        messages: [{role: "system", content: "You are a specialist in human relationships and a certified couples therapist. Your job is to answer questions and give advice, based on previously scored and annotated interactions between partners. Your response should be formatted using markdown syntax"},
                    {role: "user", content: questionTemplate}],
        max_tokens: 1024,
        temperature: 0.7
    });
    return response.data.choices[0].message.content

}

async function analyzeConversation(conversationText) {
    const promptTemplate = `
    ${analysisPrompt}
    ${conversationText}
    `;


    //const messages = [...trainingData, {role:"user", content: promptTemplate}]
    const messages = [{role:'system', content: systemInstruction},{role:"user", content: promptTemplate}]

    // The app will function with cheaper models such as gpt-4o, however the qua;ity of the advice is better with gpt-4.1
    const response = await openai.createChatCompletion({
        model: LOVEGPT_MODEL,
        messages: messages,
        temperature: 0.5
    });

    return response.data.choices[0].message.content
}

module.exports = {
    analyzeConversation, askQuestion
};

// (async () => {
//     const conversation = `[1:59 pm, 26/07/2023] Susy: EMOTIONALLY im not ok
// [2:00 pm, 26/07/2023] Susy: you dont get it Sam and is something I'm not going to share or discuss with you
// [2:00 pm, 26/07/2023] Sam Rahimi: why not?
// [2:00 pm, 26/07/2023] Sam Rahimi: don't you want me to be understanding?
// [2:00 pm, 26/07/2023] Sam Rahimi: then i need to understand!
// [2:00 pm, 26/07/2023] Susy: you are not a therapist or my therapist
// `   
//     const analysis = await analyzeConversation(conversation);
//     console.log(analysis);

//     //const advice = await getNextSteps()
//     //console.log(advice)
//     var advice = await askQuestion(analysis, "Please discuss the dynamics of this interaction and provide advice for each party if advice is needed")
//     console.log(advice)

    
//     //var alternative=  await askQuestion(analysis, "Please provide a healthier alternative for this conversation. Use the same JSON format in your answer as is used in the context")
//     //console.log(`**Healthier Alternative**
//     //${alternative}`)

//     // var sessionNotes = await askQuestion(`
//     //                                     ${analysis}
//     //                                     ${advice}`, "Please summarize the conversation, analysis, and advice into a few paragraphs, keeping the information that is important and discarding that which is not. The summary will be used to provide context when this couple comes for their next session. This should be plain text, not markdown, and should be in a format similar to the clinical notes a human therapist would make after a session - neither partner will see this summary, so it should contain an honest assessment without the need to be kind or diplomatic")
//     // console.log(`**Session Notes**
//     // ${sessionNotes}`)
// })();
