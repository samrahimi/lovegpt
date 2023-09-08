/* ui code for main lovegpt app */

window.INTAKE_DATA = null       //the unstructured data provided by the user (a text message conversation pasted in the box)
window.SESSION_CONTEXT = null   //the structured data returned by the model, consisting of each separate message and associated scoring
window.SESSION_RESULTS = null   //the user's questions about the data, and the model's responses
window.SESSION_ID = "A-" + (parseInt(Math.random() * 1000000000)).toString() //todo: uuid

const showChatMessage = (sender, content, allowUnsafe = false) => {
    if (allowUnsafe) {
        $(".chat-messages").append(`<div class="rounded p-3 mb-3 w-100 bg-neutral">
                <b>${sender}</b><br />
                ${content}    
                </div>`)
    } else {
        $(".chat-messages").append(`<div class="rounded p-3 mb-3 w-100 bg-neutral">
                    <b>${sender}</b><br />
                    ${markdown.toHTML(content)}    
                </div>`)
    }

    // Scroll to the bottom of the chatbox
    $(".chat-messages").scrollTop($(".chat-messages")[0].scrollHeight);

}

function toggleWaitScreen() {
    const factsModal = document.querySelector('facts-modal');
    const formElements = $('input, textarea, button');
    // Check if the modal is visible
    if (factsModal.isOpen()) {
        factsModal.hideModal()
        formElements.prop('disabled', false);
        console.log("closed wait screen");
    } else {
        factsModal.showModal()
        formElements.prop('disabled', true);
        console.log("opened wait screen")
    }

    // deprecated: we replaced the spinny thing with a box that displays relevant and interesting "facts" while you wait. sorry, spinny
    // const spinner = $('#wait-spinner');
    // const formElements = $('input, textarea, button');

    // if (spinner.is(':visible')) {
    //     spinner.hide();
    //     $("#wait-text").hide()
    //     formElements.prop('disabled', false);
    // } else {
    //     spinner.show();
    //     $("#wait-text").show()
    //     formElements.prop('disabled', true);
    // }
}

const askQuestion = (question, showQuestion = true) => {
    if (showQuestion)
        showChatMessage("User", question)

    //var allowUnsafe = !showQuestion
    var allowUnsafe = false

    toggleWaitScreen()
    postJson("/ask", { context: window.SESSION_CONTEXT || "Not Provided", question: question }, (response) => {
        showChatMessage("Doctor G", response, allowUnsafe)
        toggleWaitScreen()
    })
}

const getSuggestedQuestions = () => {
    askQuestion("Please provide a list of 5 questions that the couple may wish to explore with a trained therapist, either human or of an extremely advanced AI, based on the context provided. Please format your response as a markdown list", false)
    $(".query-suggestion").on("click", (event) => {
        event.preventDefault()
        askQuestion($(this).html(), true)
    })
}


const showResults = (conversationData) => {
    window.SESSION_CONTEXT = JSON.parse(conversationData)
    showChatMessage("System", `Loaded analysis into chatbot, ready to chat. Don't know what to say? Click the button for personalized suggestions!<br /><br /><button class="btn btn-secondary w-100" id="getSuggestions" type="button">Get Suggestions</button>`, true)
    for (let message of window.SESSION_CONTEXT) {
        let alignment = (message.turn === "user") ? "start" : "end";

        //color code the messages based on toxicity
        let bgColor = `bg-${message.health_score} text-white`;

        let chatBubble = `
                <div class="d-flex flex-column align-items-${alignment} mb-4">
                    <div class="rounded p-3 w-100 ${bgColor}">
                        ${message.message_text}
                        <br /><br />
                        <small>
                        <b>Emotional Tone:</b> ${message.emotional_tone_description}
                    </small><br />
                    <small>
                        <b>Health Score:</b> ${message.health_score}
                    </small><br />
                    <small>
                        <b>Remarks:</b> ${message.commentary}
                    </small>

                    </div>
                </div>
            `;

        $("#chat-container").append(chatBubble);
    }
    $("#getSuggestions").on("click", (event) => {
        getSuggestedQuestions()
    });
}
const postJson = (route, payload, callback) => {
    $.ajax({
        url: route,
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(payload),
        success: callback
    })
}

const getAdvice = () => {
    //Get advice using the analysis as context
    console.log("Requesting overall advice")
    $.ajax({
        url: '/advice',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ context: window.SESSION_CONTEXT }),
        success: function (response) {
            console.log("advice received")
            console.log(JSON.stringify(response))
            window.SESSION_RESULTS = markdown.toHTML(response)
            $("#advice").html(window.SESSION_RESULTS)
        }
    })

}

$(document).ready(function () {
    $(".chat-toggle").on("click", () => {

        if (!window.SESSION_CONTEXT && !confirm("For best results,  provide some context first (i.e. texts from partner). Continue anyway?")) 
                return false
        $(".sidebar-container").show()
    })
    $("#hamster-removal").on("click", () => {
        if (window.SESSION_CONTEXT != null) {
            askQuestion("I am experiencing severe anxiety and insecurity regarding this situation. Tell me what I can do to feel less anxious right away, based on the context and on your knowledge of CBT, DBT, family therapy, Eastern meditation-based practices...", false)
        }
        else {
            askQuestion("I am experiencing severe anxiety and insecurity regarding a romantic situation. It is unclear whether these feelings are justified or not. Tell me what I can do to feel less anxious right away, based on your knowledge of CBT, DBT, family therapy, Eastern meditation-based practices...", false)
        }

        showChatMessage("System", "The doctor will be with you in a moment... Dr. G is trained in advanced psychotherapy techniques proven to reduce anxiety")
        $(".sidebar-container").show()

    })
    $("#sendChatMessage").on("click", () => {
        if ($("#inputBox").val().trim().length > 2000) {
            $("#inputBox").val("")
            alert("Please send a shorter message")
        } else {
            askQuestion($("#inputBox").val().trim())
            $("#inputBox").val("")
        }
    })
    // Get the input element by its ID
    const inputBox = document.getElementById("inputBox");

    // Add a keydown event listener
    inputBox.addEventListener("keydown", function (event) {
        // Check if the Enter key was pressed
        if (event.key === "Enter" || event.keyCode === 13) {
            // Prevent the default action, if needed
            event.preventDefault();

            // Your action here
            console.log("Enter key was pressed, sending message");

            $("#sendChatMessage").click()
        }
    });


    $(".close-sidebar-button").on("click", () => { $(".sidebar-container").hide() })

    $('#analyze-form').submit(function (event) {
        event.preventDefault();
        window.INTAKE_DATA = $("#conversation").val()
        toggleWaitScreen()
        const conversationContent = $('#conversation').val();
        const payload = {
            input: conversationContent
        };

        $.ajax({
            url: '/analyze',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(payload),
            success: function (response) {
                console.log(JSON.stringify(response))
                toggleWaitScreen()
                $("#intake-screen").hide()
                $("#main-screen").show()
                showResults(response)
                getAdvice()
            },
            error: function (error) {
                console.log('Error:', error);
                alert('Analysis failed. For best results, select the relevant texts from your messaging app, copy, and paste into the box.');
            }
        });
    });
});

