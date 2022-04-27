import PageTemplate from './Template'
import '../css/About.css';

function App() {
    return (
        <>
            <PageTemplate highLight="1" />

            <div className='AboutPage'>

                <ul className="ProfileStatsBox">
                    <h1>About Us</h1>
                    <div className="ProfileStatsBoxGroup">
                        <h2>Who We Are</h2>
                        <li>This was done as a group project for ITCS 4152/5152 at University of North Carolina at Charlotte.</li>
                    </div>
                    <div className="ProfileStatsBoxGroup">
                        <h2>Thomas Carr</h2>
                        <li>Demo Website Front/Backend</li>
                        <li>Model Creation/Research</li>
                        <li>Collection/Organization of Data</li>
                    </div>

                    <div className="ProfileStatsBoxGroup">
                        <h2>Kyle Ward</h2>
                        <li>Model Creation/Research</li>
                        <li>Collection/Organization of Data</li>
                        <li>Presentation Creation</li>
                    </div>

                    <div className="ProfileStatsBoxGroup">
                        <h2>Jay Yadav</h2>
                        <li>Demo Website Front/Backend</li>
                        <li>Preperation of Data/Research</li>
                        <li>Presentation Creation</li>
                    </div>

                    <div className="ProfileStatsBoxGroup">
                        <h2>Payne Miller</h2>
                        <li>Assisted of Demo Backend</li>
                        <li>Preperation of Data/Research</li>
                        <li>Presentation Creation</li>
                    </div>

                    <div className="ProfileStatsBoxGroup" style={{ marginBottom: '1rem' }}>
                        <h2>Makaila Vang</h2>
                        <li>Preperation of Data/Research</li>
                        <li>Presentation Creation</li>
                    </div>
                </ul>
            </div>
        </>
    );
}

export default App;
