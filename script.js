// =========================================================
// ENHANCED SCRIPT.JS — ML CONNECTED + KEYWORD SAFETY + IMAGES
// =========================================================

document.addEventListener('DOMContentLoaded', function() {

    const MODEL_API_URL = "http://127.0.0.1:5000/api/predict";

    // === YOUR ORIGINAL LISTS (PRESERVED EXACTLY) ===
    const hateKeywords = [
        'kill', 'attack','fuck','destroy', 'exterminate', 'annihilate', 'eliminate',
        'hate', 'violence', 'terrorist', 'racist', 'nazi', 'supremacist',
        'genocide', 'ethnic cleansing', 'lynch', 'burn', 'destroy','fucker','bitch',
        'white power', 'black power', 'racial purity', 'ethnic purity',
        'jihad', 'crusade', 'holy war', 'infidel', 'heretic', 'apostate',
        'fascist', 'extremist', 'radical', 'zealot', 'bigot', 'xenophobe','mulla','atankwadi',
        'katua', 'mullah', 'jihadi', 'atankvadi', 'jihadi katua', 'mleccha', 'kafir', 'murti pujak', 'beiman',
        'chamar', 'bhangi', 'dhed', 'mochi', 'kori', 'khatik', 'balmiki', 'pasi',
        'upwala', 'madrasi','randi', 'veshya', 'kamin', 'badchalan', 'charitraheen', 'naqli',
        'maro salo ko', 'kat do unko', 'jala do', 'bhagao enko', 'desh chodo', 'wapis jao', 'tukde tukde gang', 'deshdrohi',
        'gaddar','awara', 'nalayak', 'nikamma', 'kabari', 'kanjar', 'rakshas', 'shaitan', 'danav',
        'kala', 'kali','lulla','andh bhakt', 'presstitue', 'sickular', 'librandu', 'urban naxal', 'tukde tukde',
        'chhakka', 'hijda', 'napunsak', 'samlaingik','nigga',
        'should be killed', 'support attacks', 'cleanse our nation', 'forced out', 'burn down',
        'drive them out', 'don\'t deserve to live', 'eliminate all', 'death to all', 'traitors to the nation',
        'physically remove', 'invaders from our land', 'support violence', 'waste of space', 'eliminated from this country',
        'animals who', 'civilized people', 'places of worship', 'by any means necessary', 'belong here',
        'should be forced', 'should be killed', 'terrorists and'
    ];

    const offensiveKeywords = [
        'stupid', 'idiot', 'moron', 'dumb', 'ugly', 'fat', 'loser', 'retard',
        'bastard', 'bitch', 'whore', 'slut', 'asshole', 'dick', 'pussy', 'fuck',
        'shit', 'cunt', 'motherfucker', 'douchebag', 'scumbag', 'shithead',
        'faggot', 'dyke', 'tranny', 'retard', 'cripple', 'midget', 'spastic',
        'nigger', 'chink', 'spic', 'kike', 'gook', 'wetback', 'raghead', 'towelhead',
        'madarchod', 'behenchod', 'bhosdike', 'chutiya','taka', 'gandu', 'lund',
        'kutta', 'kamine', 'harami', 'suar', 'khotey', 'randi', 'rand',
        'kamlina', 'besharm', 'badmas', 'awara', 'nalayak','chut', 'nikamma',
        'kabari', 'chamar', 'bhangi', 'kanjar', 'veshya', 'raand','chod',
        'kala', 'kali', 'kallu', 'bhootni', 'shaitan', 'rakshas','kalua','tatti',
        'jhatu','dedh','hapsi','napunsak','loda','bhosda',
        'waste of space and oxygen','black people'
    ];

    // === EXTRA FAIL-PROOF KEYWORDS ADDED (WITHOUT REMOVING YOURS) ===
    const extraSafetyPhrases = [
        // Explicit threats & promotion
        "i will kill", "i will attack", "be shot", "be hanged", "hang them", "shoot them",
        "rape them", "i support killing", "i support attacking", "we should beat them",
        "beat them up", "stone them", "stoniṅg", "st0ne them",

        // Implicit hate
        "go to hell", "rot in hell", "you are a disease", "scum of society",
        "waste of oxygen", "waste of air", "should not have rights",
        "not real indians", "not real citizens", "banish them", "erase them",

        // Sarcastic hate patterns
        "oh look another one", "as expected from them", "of course they would",
        "these people again",

        // Mixed / Hinglish & phonetic variants
        "salo ko maro", "inko maro", "unko kato", "suar log", "kutta log",
        "g@ndu", "gandu log", "madrasi log", "go back log", "go back",

        // Character obfuscation (social media evasion)
        "k!ll", "k1ll", "r@ndi", "randi log", "b!tch", "b1tch", "sh1t", "sh!t",
        "f@ck", "f#ck", "n1gger", "n!gger", "t3rrorist", "r4cist",

        // Long phrases
        "should be sent back", "should be thrown out", "throw them out",
        "they are a stain", "stain on society", "drag them out", "drag them out of the country"
    ];

    // FINAL MASTER LIST (ORIGINAL + SAFETY)
    const masterKeywords = [...hateKeywords, ...offensiveKeywords, ...extraSafetyPhrases];

    function detectMasterKeywords(text) {
        const found = new Set();
        const lower = text.toLowerCase();
        masterKeywords.forEach(k => {
            if (lower.includes(k.toLowerCase())) found.add(k);
        });
        return Array.from(found);
    }

    // Initialize charts
    const performanceChart = new Chart(document.getElementById('performanceChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: ["Random Forest", "SVM", "Naïve Bayes", "KNN", "Decision Tree"],
            datasets: [{
                label: "Accuracy (%)",
                data: [90.33, 90.44, 86.40, 63.28, 87.55],
                backgroundColor: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                }
            }
        }
    });

    const featureChart = new Chart(document.getElementById('featureChart').getContext('2d'), {
        type: 'radar',
        data: {
            labels: ["Accuracy", "Precision", "Recall", "F1-Score", "Speed"],
            datasets: [
                {
                    label: "Bigram TF-IDF",
                    data: [90, 88, 90, 89, 70],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.2)'
                },
                {
                    label: "Unigram TF-IDF",
                    data: [85, 83, 84, 83, 85],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.2)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            }
        }
    });

    const networkChart = new Chart(document.getElementById('networkChart').getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: "User Accounts",
                data: [
                    {x: 20, y: 30},
                    {x: 40, y: 10},
                    {x: 15, y: 25},
                    {x: 35, y: 40},
                    {x: 50, y: 20}
                ],
                backgroundColor: '#3498db',
                pointRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Interaction Frequency'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Content Severity Score'
                    }
                }
            }
        }
    });

    // === ML CONNECTED FORM LOGIC ===
    document.getElementById('detectionForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const textInput = document.getElementById('textInput').value;
        const algo = document.getElementById('algorithmSelect').value;
        const featureType = document.getElementById('featureSelect').value;

        if (!textInput.trim()) {
            alert("Please enter text to analyze.");
            return;
        }

        const badge = document.getElementById('classificationBadge');
        badge.textContent = "Analyzing...";
        badge.className = "classification-badge";

        // Show loading state
        document.getElementById('resultsContainer').style.display = "block";
        document.querySelector('#resultsContainer .card-body').style.opacity = "0.7";

        try {
            const res = await fetch(MODEL_API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    text: textInput, 
                    algorithm: algo,
                    feature_type: featureType 
                })
            });

            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }

            const data = await res.json();
            if (!data.success) throw new Error(data.error || "Model API error");

            let classification = data.result.classification;
            let confidence = data.result.confidence;

            // Enhanced keyword fallback
            const hits = detectMasterKeywords(textInput);
            const hasHateKeyword = hits.some(k => hateKeywords.includes(k));
            const hasOffensiveKeyword = hits.some(k => offensiveKeywords.includes(k));

            // Override ML prediction if keywords strongly indicate hate/offensive
            if (classification === "Neutral" && hits.length > 0) {
                if (hasHateKeyword) {
                    classification = "Hate Speech";
                    // Adjust confidence based on keyword presence
                    confidence = { hate: 85, offensive: 10, neutral: 5 };
                } else if (hasOffensiveKeyword) {
                    classification = "Offensive Language";
                    confidence = { hate: 15, offensive: 75, neutral: 10 };
                }
            }

            // Update UI
            updateResults(classification, confidence, hits, data.algorithm || 'selected model');

        } catch(err) {
            console.error("Prediction error:", err);
            // Fallback to keyword-based analysis if API fails
            const hits = detectMasterKeywords(textInput);
            const hasHateKeyword = hits.some(k => hateKeywords.includes(k));
            const hasOffensiveKeyword = hits.some(k => offensiveKeywords.includes(k));
            
            let classification, confidence;
            
            if (hasHateKeyword) {
                classification = "Hate Speech";
                confidence = { hate: 85, offensive: 10, neutral: 5 };
            } else if (hasOffensiveKeyword) {
                classification = "Offensive Language";
                confidence = { hate: 15, offensive: 75, neutral: 10 };
            } else {
                classification = "Neutral";
                confidence = { hate: 5, offensive: 10, neutral: 85 };
            }
            
            updateResults(classification, confidence, hits, "keyword analysis (API unavailable)");
        }
    });

    function updateResults(classification, confidence, hits, algorithmUsed) {
        const badge = document.getElementById('classificationBadge');
        badge.textContent = classification;
        badge.className = `classification-badge ${classification.toLowerCase().replace(/ /g, "-")}`;

        // Update progress bars
        document.getElementById('hateProgress').style.width = `${confidence.hate}%`;
        document.getElementById('hateProgress').textContent = `Hate Speech: ${confidence.hate.toFixed(1)}%`;

        document.getElementById('offensiveProgress').style.width = `${confidence.offensive}%`;
        document.getElementById('offensiveProgress').textContent = `Offensive: ${confidence.offensive.toFixed(1)}%`;

        document.getElementById('neutralProgress').style.width = `${confidence.neutral}%`;
        document.getElementById('neutralProgress').textContent = `Neutral: ${confidence.neutral.toFixed(1)}%`;

        // Update keywords list
        const list = document.getElementById('keywordsList');
        list.innerHTML = "";
        if (hits.length > 0) {
            hits.slice(0, 10).forEach(k => { // Show max 10 keywords
                const span = document.createElement("span");
                span.className = "badge bg-secondary";
                span.textContent = k;
                list.appendChild(span);
            });
            if (hits.length > 10) {
                const moreSpan = document.createElement("span");
                moreSpan.className = "badge bg-info";
                moreSpan.textContent = `+${hits.length - 10} more`;
                list.appendChild(moreSpan);
            }
        } else {
            list.innerHTML = '<span class="text-muted">No specific keywords detected</span>';
        }

        // Update explanation
        document.getElementById('explanationText').textContent =
            `Using ${algorithmUsed}, this text was classified as "${classification}". ` +
            `${hits.length > 0 ? `Detected ${hits.length} relevant keywords.` : 'No strong keyword indicators found.'}`;

        // Remove loading state
        document.querySelector('#resultsContainer .card-body').style.opacity = "1";

        // Smooth scroll to results
        document.getElementById('resultsContainer').scrollIntoView({ behavior: "smooth" });
    }

    // Check API health on load
    async function checkAPIHealth() {
        try {
            const res = await fetch('http://127.0.0.1:5000/api/health');
            const data = await res.json();
            console.log('API Health:', data);
        } catch (err) {
            console.warn('API not reachable:', err.message);
        }
    }
    
    checkAPIHealth();
});