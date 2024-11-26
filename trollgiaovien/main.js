const button = document.querySelector("button");
const header = document.querySelector("h1");
const graphic = document.querySelector("div");

const actions = [
	{
		text: "NOPE!!",
		move: "translate(80%, -30%)"
	},
	{
		text: "heehee",
		move: "translate(-100%, -70%)"
	},
	{
		text: "<span class='material-symbols-outlined'>sentiment_excited</span>",
		move: "translate(120%, 80%)"
	},
	{
		text: "<span class='material-symbols-outlined'>play_arrow</span>",
		move: "translate(0, 0)"
	},
	{
		text: "??",
		move: "translate(-80%, 100%)"
	},
	{
		text: "...",
		move: "translate(10%, -150%)"
	}
];

const getRandomInt = (max) => {
	return Math.floor(Math.random() * max);
};

button.addEventListener("mouseover", () => {
	index = getRandomInt(actions.length);
	if (button.classList.contains("play")) {
		button.innerHTML = actions[index].text;
		button.style.transform = actions[index].move;
	}
});

button.addEventListener("click", () => {
	if (button.classList.contains("play")) {
		button.classList.remove("play");
		button.classList.add("pause");
		button.style.transform = "translate(0,0)";

		graphic.classList.add("dot");

		button.innerHTML = "<span class='material-symbols-outlined'>pause</span>";
		header.innerHTML = "Press pause";
	} else if (button.classList.contains("pause")) {
		button.classList.remove("pause");
		button.classList.add("play");
		button.style.transform = "translate(0,0)";

		graphic.classList.remove("dot");

		button.innerHTML =
			"<span class='material-symbols-outlined'>play_arrow</span>";
		header.innerHTML = "Press play... if you can";
	}
});
