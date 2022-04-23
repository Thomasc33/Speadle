import React from 'react'
import Particles from 'react-tsparticles'

const particles = () => {
    return (
        <Particles
            width='100vw'
            height='100vh'
            options={{
                background: {
                    color: {
                        value: "#2C2F33"
                    },
                },
                fpsLimit: 60,
                interactivity: {
                    detectsOn: "window",
                    events: {
                        onHover: {
                            enable: true,
                            mode: [],
                            parallax: {
                                enable: true,
                                force: 20,
                                smooth: 10,
                            }
                        }
                    }
                },
                particles: {
                    number: {
                        value: 60,
                    },
                    color: {
                        value: "#03cafc"
                    },
                    links: {
                        color: "#99AAB5",
                        distance: 250,
                        enable: true,
                        opacity: .5,
                        width: 1
                    },
                    move: {
                        enable: true,
                        speed: .3,
                        direction: 'random',
                        random: true
                    },
                }
            }}
        />
    )
}

export default particles