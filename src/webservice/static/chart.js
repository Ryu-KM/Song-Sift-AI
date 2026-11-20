document.addEventListener("DOMContentLoaded", function () {
  // Function to extract song data from the page
  function getSongData() {
    let songItems = document.querySelectorAll(".song-item"); // Select all song elements
    let titles = [];
    let scores = [];

    songItems.forEach((item) => {
      let titleElement = item.querySelector(".top-song-title");
      let scoreElement = item.querySelector(".top-song-score");

      if (titleElement && scoreElement) {
        let title = titleElement.textContent.trim();
        let scoreText = scoreElement.textContent.replace("Score:", "").trim();
        let score = parseFloat(scoreText);

        if (!isNaN(score)) {
          titles.push(title);
          scores.push(score);
        }
      }
    });

    console.log("Extracted Titles:", titles);
    console.log("Extracted Scores:", scores);

    return { titles, scores };
  }

  // Call function to get the data
  let { titles, scores } = getSongData();

  // Error handling if no data is extracted
  if (titles.length === 0 || scores.length === 0) {
    console.warn("No valid data extracted for chart!");
    return;
  }

  // Dynamic y-axis range based on min/max scores
  let minScore = Math.min(...scores);
  let maxScore = Math.max(...scores);
  let rangePadding = (maxScore - minScore) * 0.1;
  let yMin = Math.max(0, minScore - rangePadding);
  let yMax = maxScore + rangePadding;

  // Create a bar chart using Chart.js
  let ctx = document.getElementById("myChart")?.getContext("2d");

  if (ctx) {
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: titles,
        datasets: [
          {
            label: "Similarity Score",
            data: scores,
            backgroundColor: "rgba(55, 0, 179, 0.7)",
            borderColor: "rgba(55, 0, 179, 1)",
            borderRadius: 20,
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Song Title" },
            ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 },
          },
          y: {
            beginAtZero: false,
            min: yMin,
            max: yMax,
            title: { display: true, text: "Similarity Score" },
          },
        },
      },
    });
  } else {
    console.error("Chart canvas not found!");
  }
});
