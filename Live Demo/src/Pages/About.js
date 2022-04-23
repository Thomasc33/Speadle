import PageTemplate from './Template'
import '../css/About.css';

function App() {
    return (
        <>
            <PageTemplate highLight="1" />
            <ul className="ProfileStatsBox">
                <h1>About Us</h1>
                <div className="ProfileStatsBoxGroup">
                    <h2>Who We Are</h2>
                    <li>This was done as a group project for ITCS 4152/5152 at University of North Carolina at Charlotte. *Add more here*</li>
                </div>
                <div className="ProfileStatsBoxGroup">
                    <h2>Thomas Carr</h2>
                    <li>Demo Website Front/Backend</li>
                    <li>Model Creation/Research</li>
                    <li>Collection/Organization of Data</li>
                </div>
                <h1> </h1>
                <h1> </h1>
            </ul>
        </>
    );
}

export default App;
