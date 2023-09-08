const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors')
const app = express();
const port = 3000;
const analyzer = require("./analyzer.js")
const path = require("path")
// Middleware to parse JSON payloads
app.use(bodyParser.json());
app.use(cors())

app.use(express.static(path.join(__dirname, 'public')));
// Define a route for the default document
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
  
// /analyze endpoint
app.post('/analyze', async(req, res) => {
  const data = req.body;
  const result = await analyzer.analyzeConversation(data.input)

  res.json(result);
});

//advice endpoint (preset query)
app.post("/advice", async(req, res) =>  {
    const data = req.body
    var advice = await analyzer.askQuestion(data.context, "Please discuss the dynamics of this interaction and provide advice for each party if advice is needed")
    res.json(advice)
})

// /ask endpoint
app.post('/ask', async(req, res) => {
  const data = req.body;

  // Perform some query logic here
  const result = await analyzer.askQuestion(data.context, data.question)

  res.json(result);
});

// Start the server
app.listen(port, () => {
  console.log(`API running on http://localhost:${port}`);
});
