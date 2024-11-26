class RandomCoordinates {
    constructor() {
        this.x = Math.random() * 100 + '%';
        this.y = Math.random() * 100 + '%';
    }
}


class ShootingStar {
    constructor() {
        this.star = document.createElement('div');
        this.star.classList.add('star');

        this.start = new RandomCoordinates();

        this.setStyles();
    }

    setStyles() {
        this.setNewStyleVar('--start-x', this.start.x);
        this.setNewStyleVar('--start-y', this.start.y);

        this.setNewStyleVar('--duration', Math.random() * 6 + 5 + 's'); 

        this.setNewStyleVar('--delay', Math.random() + 's');

        this.setNewStyleVar('--top', Math.random() * 100 - 30 + '%');
        this.setNewStyleVar('--left', Math.random() * 100 - 30+ '%');

        this.setNewStyleVar('--size', Math.random() + 0.5);

        this.setNewStyleVar('--angle', Math.random() * 360 + 'deg');
    }

    setNewStyleVar(styleName, styleValue) {
        this.star.style.setProperty(styleName, styleValue);
    }

    getStar() {
        return this.star;
    }
}

new ShootingStar();

class NightSky {
    constructor() {
        this.nightSky = document.getElementById('night-sky');

        this.numberOfStars = Number.parseInt(this.nightSky.style.getPropertyValue('--number-of-stars'));
        this.addStars(this.numberOfStars);
    }

    addStars(numberOfStars) {
        for (let i = 0; i < numberOfStars; i++) {
            this.nightSky.appendChild(new ShootingStar().getStar());
        }
    }
}

new NightSky();
