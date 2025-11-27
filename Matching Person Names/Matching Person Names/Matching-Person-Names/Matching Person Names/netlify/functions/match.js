const NAMES = [
    "Geetha", "Geeta", "Gita", "Gitu", "Githa", "Geethika", "Geeti", "Gita Devi",
    "Rama", "Ramu", "Ramesh", "Ramya", "Ramana", "Raman", "Rama Devi", "Ramu Sharma",
    "Sita", "Seetha", "Sitha", "Sita Devi", "Sita Ram", "Sheetal",
    "Krishna", "Krish", "Krisna", "Krishnan", "Krishna Kumar", "Krishna Devi", "Krish Patel",
    "Anita", "Anitha", "Anita Devi", "Anita Kumari", "Anita Sharma", "Anita Reddy",
    "Latha", "Lata", "Latika", "Latha Devi", "Latha Kumari", "Lakshmi",
    "Sunita", "Sunitha", "Suni", "Suneeta", "Sunita Devi", "Sunita Kumari", "Sudev", "Sudevi",
    "Kiran", "Kiron", "Kirana", "Kiran Kumar", "Kiran Devi", "Kiran Patel",
    "Priya", "Priyanka", "Priya Kumari", "Priya Devi", "Priyanka Sharma",
    "Meena", "Meenakshi", "Meenu", "Meena Kumari", "Meena Devi",
    "Radha", "Radhika", "Radha Devi", "Radhika Sharma", "Radhika Kumari",
    "Shiva", "Shivani", "Shiva Kumar", "Shivani Devi", "Shankar",
    "Lakshmi", "Lakshmi Devi", "Laxmi", "Lakshmi Kumari", "Lakshmi Narayan",
    "Sarita", "Saritha", "Sarita Devi", "Sarita Kumari", "Sarita Sharma",
    "Kavitha", "Kavya", "Kavita", "Kavitha Devi", "Kavitha Kumari",
    "Deepa", "Devi", "Devika", "Deepa Devi", "Deepa Kumari", "Devi Priya",
    "Nithya", "Nitya", "Neethu", "Nithya Devi", "Nitya Kumari",
    "Swathi", "Swati", "Swapna", "Swathi Devi", "Swati Kumari",
    "Divya", "Devi Priya", "Devi Sree", "Divya Devi", "Divya Kumari",
    "Madhavi", "Madhu", "Madhura", "Madhavi Devi", "Madhavi Kumari",
    "Bhavani", "Bhavana", "Bhavitha", "Bhavani Devi", "Bhavana Kumari",
    "Sandhya", "Sindhu", "Sandeep", "Sandhya Devi", "Sindhu Kumari",
    "Roja", "Rupa", "Rupali", "Roja Devi", "Rupa Kumari",
    "Jyothi", "Jyoti", "Jyotsna", "Jyothi Devi", "Jyoti Kumari",
    "Indira", "Indu", "Indra", "Indira Devi", "Indira Kumari",
    "Chitra", "Chithra", "Chandrika", "Chitra Devi", "Chithra Kumari",
    "Malini", "Meena Kumari", "Meera", "Malini Devi", "Meera Devi",
    "Pooja", "Puja", "Pushpa", "Pooja Devi", "Puja Kumari",
    "Rekha", "Renu", "Revathi", "Rekha Devi", "Renu Kumari",
    "Amit", "Amita", "Amitabh", "Aishwarya", "Akash", "Anil", "Anjali", "Arjun",
    "Deepak", "Dinesh", "Divakar", "Esha", "Farhan", "Gayatri", "Harish", "Isha",
    "Kavita", "Manoj", "Neha", "Prakash", "Praveen", "Rajesh", "Rakesh", "Rashmi",
    "Sanjay", "Sandeep", "Santosh", "Shankar", "Shantanu", "Shashi", "Shilpa", "Sonali",
    "Sourabh", "Srinivas", "Subhash", "Sudhir", "Sujata", "Sukanya", "Sumit", "Sunil",
    "Supriya", "Suraj", "Suresh", "Tanvi", "Tanuja", "Tejas", "Uma", "Umesh",
    "Upendra", "Vikas", "Vikram", "Vimal", "Vinay", "Vineet", "Vishal", "Vivek"
];

function preprocess(name) {
    return name.trim().toLowerCase();
}

function levenshtein(a, b) {
    const m = a.length;
    const n = b.length;
    if (m === 0) return n;
    if (n === 0) return m;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
            dp[i][j] = Math.min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            );
        }
    }
    return dp[m][n];
}

function similarity(a, b) {
    const dist = levenshtein(a, b);
    const maxLen = Math.max(a.length, b.length) || 1;
    return 1 - dist / maxLen;
}

exports.handler = async (event) => {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: JSON.stringify({ error: 'Method not allowed' }) };
    }
    try {
        const data = JSON.parse(event.body || '{}');
        const query = preprocess(String(data.name || ''));
        if (!query) {
            return { statusCode: 400, body: JSON.stringify({ error: 'Please provide a name to search' }) };
        }
        const scores = NAMES.map((name) => {
            const s = similarity(query, preprocess(name));
            let boost = s;
            const nl = name.toLowerCase();
            if (nl.includes(query) || query.includes(nl)) boost *= 1.1;
            return { name, score: boost };
        });
        scores.sort((a, b) => b.score - a.score);
        const best = scores[0];
        const top = scores.slice(0, 10);
        return {
            statusCode: 200,
            body: JSON.stringify({
                best_match: { name: best.name, score: Number(best.score) },
                top_matches: top.map(t => ({ name: t.name, score: Number(t.score) }))
            })
        };
    } catch (e) {
        return { statusCode: 500, body: JSON.stringify({ error: 'Server error' }) };
    }
};
