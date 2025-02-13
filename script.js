const form = document.getElementById('login-form');
const codeInput = document.getElementById('code');
const resultDiv = document.getElementById('result');

const secretCode = 'poplu'; // Replace with your secret code
const nextPageUrl = 'main.html'; // Redirect URL

form.addEventListener('submit', (e) => {
    e.preventDefault(); // Prevent form from refreshing
    const userInput = codeInput.value.trim(); // Get user input

    if (userInput === secretCode) {
        resultDiv.innerHTML = 'Mera popluuu';
        setTimeout(() => {
            window.location.href = nextPageUrl; // Redirect after 1 sec
        }, 1000);
    } else {
        resultDiv.innerHTML = 'mai sirf poplush ki najuka hun';
    }
});
