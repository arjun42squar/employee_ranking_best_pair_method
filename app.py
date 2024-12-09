from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import random
import numpy as np
import webbrowser

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Add this to disable caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Sample data input
data = {
    'Employee': ['Jijo', 'Jobin', 'Suhair', 'Anakha', 'Ponny', 'Aneesh R', 'Colin', 'Yoosef', 'Shinoj', 'Arjun', 'Najmal', 'Sreelakshmi', 'Anand', 'Raghu', 'Nithin', 'Rijo', 'Hari', 'Jimmy', 'Mansoor', 'Fredy', 'Bejoy', 'Rahul KD', 'Sasnesh'],
    'Job_Level': ['Senior', 'Mid-Level', 'Senior', 'Junior', 'Mid-Level', 'Senior', 'Junior', 'Mid-Level', 'Senior', 'Junior', 'Junior', 'Mid-Level', 'Mid-Level', 'Senior', 'Senior', 'Senior', 'Senior', 'Mid-Level', 'Mid-Level', 'Mid-Level', 'Mid-Level', 'Junior', 'Junior'],
    'Experience': [17, 7, 16, 1, 5, 10, 2, 9, 14, 3, 2, 6, 7, 14, 15, 12, 14, 9, 8, 8, 9, 3, 2],
    'Location': ['Utah', 'Kochi', 'Utah', 'Utah', 'Kochi', 'Utah', 'Utah', 'Kochi', 'Kochi', ' Kochi', 'Kochi', 'Utah', 'Kochi', 'Utah', 'Kochi', 'Utah', 'Kochi', 'Utah', 'Kochi', 'Utah', 'Utah', 'Kochi', 'Utah']
}
df = pd.DataFrame(data)

class EmployeeRanking:
    def __init__(self, dataframe):
        self.df = dataframe
        self.reset()

    def reset(self):
        """Reset all comparison and ranking data"""
        self.rankings = {}
        self.current_pairs = []
        self.current_pair_index = 0
        self.completed_comparisons = set()
        self.wins = {}
        self.losses = {}
        print("Rankings reset. Ready for new comparisons.")

    def get_next_pair(self):
        while self.current_pair_index < len(self.current_pairs):
            pair = self.current_pairs[self.current_pair_index]
            emp1, emp2 = pair
            
            # Create a unique key for this comparison (order-independent)
            comparison_key = tuple(sorted([emp1['Employee'], emp2['Employee']]))
            
            # Skip if comparing same employee (this shouldn't happen, but let's check)
            if emp1['Employee'] == emp2['Employee']:
                print(f"Warning: Skipping self-comparison for {emp1['Employee']}")
                self.current_pair_index += 1
                continue
            
            # Skip if this comparison has already been done
            if comparison_key in self.completed_comparisons:
                print(f"Skipping duplicate comparison: {emp1['Employee']} vs {emp2['Employee']}")
                self.current_pair_index += 1
                continue
            
            # Skip if we can determine the outcome based on previous comparisons
            if (emp2['Employee'] in self.wins[emp1['Employee']] or 
                emp1['Employee'] in self.wins[emp2['Employee']]):
                print(f"Skipping transitive comparison: {emp1['Employee']} vs {emp2['Employee']}")
                self.current_pair_index += 1
                continue
                
            self.current_pair_index += 1
            print(f"\nPresenting comparison {len(self.completed_comparisons) + 1}: {emp1['Employee']} vs {emp2['Employee']}")
            return pair
            
        # If we've exhausted all pairs, return None to trigger rankings display
        print("No more comparisons needed. Ready to show rankings.")
        return None

    def record_comparison(self, winner_id, loser_id, job_level):
        # Add to completed comparisons
        comparison_key = tuple(sorted([winner_id, loser_id]))
        if comparison_key in self.completed_comparisons:
            print(f"Warning: Duplicate comparison detected - {winner_id} vs {loser_id}")
            return
        
        self.completed_comparisons.add(comparison_key)
        print(f"\nRecording comparison {len(self.completed_comparisons)}: {winner_id} beat {loser_id}")
        
        if job_level not in self.rankings:
            self.rankings[job_level] = {}
            # Initialize all employees in this job level with 0 wins
            for emp in self.wins.keys():
                if self.df[self.df['Employee'] == emp]['Job_Level'].iloc[0] == job_level:
                    self.rankings[job_level][emp] = 0
        
        # Record direct win/loss
        self.wins[winner_id].add(loser_id)
        self.losses[loser_id].add(winner_id)
        
        # Apply transitive property
        for indirect_loser in self.wins[loser_id]:
            self.wins[winner_id].add(indirect_loser)
            self.losses[indirect_loser].add(winner_id)
            self.completed_comparisons.add(tuple(sorted([winner_id, indirect_loser])))
            print(f"Transitive win: {winner_id} automatically beats {indirect_loser} (because they lost to {loser_id})")
        
        # Update rankings based on total wins
        self.rankings[job_level][winner_id] = len(self.wins[winner_id])
        self.rankings[job_level][loser_id] = len(self.wins[loser_id])
        
        print(f"\nCurrent rankings for {job_level}:")
        sorted_rankings = sorted(self.rankings[job_level].items(), key=lambda x: x[1], reverse=True)
        for emp, wins in sorted_rankings:
            print(f"  {emp}: {wins} wins")
            print(f"    Wins against: {self.wins[emp]}")
            print(f"    Losses to: {self.losses[emp]}")

    def filter_employees(self, job_level=None, location=None, min_experience=None):
        filtered_df = self.df.copy()
        
        # First level filtering
        if location:
            filtered_df = filtered_df[filtered_df['Location'] == location]
        if min_experience is not None:
            filtered_df = filtered_df[filtered_df['Experience'] >= min_experience]
            
        # Create experience bands within each job level
        def get_experience_band(exp, job_level):
            if job_level == 'Senior':
                if exp >= 15: return 'Senior-High'
                if exp >= 10: return 'Senior-Mid'
                return 'Senior-Low'
            elif job_level == 'Mid-Level':
                if exp >= 8: return 'Mid-High'
                if exp >= 5: return 'Mid-Mid'
                return 'Mid-Low'
            else:  # Junior
                if exp >= 3: return 'Junior-High'
                if exp >= 1: return 'Junior-Mid'
                return 'Junior-Low'
        
        # Add experience band column
        filtered_df['Experience_Band'] = filtered_df.apply(
            lambda row: get_experience_band(row['Experience'], row['Job_Level']), 
            axis=1
        )
        
        # Sort by experience within each band
        filtered_df = filtered_df.sort_values(['Experience_Band', 'Experience'], ascending=[True, False])
        
        return filtered_df

    def prepare_clustering(self, filtered_df):
        self.reset()
        
        # Initialize tracking dictionaries
        self.wins = {emp['Employee']: set() for emp in filtered_df.to_dict('records')}
        self.losses = {emp['Employee']: set() for emp in filtered_df.to_dict('records')}
        
        same_level_pairs = []
        
        # First, group by Job Level
        for job_level in filtered_df['Job_Level'].unique():
            job_level_df = filtered_df[filtered_df['Job_Level'] == job_level]
            
            # Then, subgroup by Location within each Job Level
            for location in job_level_df['Location'].unique():
                location_df = job_level_df[job_level_df['Location'] == location]
                
                if len(location_df) > 1:
                    # Sort employees by experience within same location
                    employees = location_df.sort_values('Experience', ascending=False).to_dict('records')
                    
                    # Compare adjacent employees by experience
                    for i in range(len(employees)-1):
                        same_level_pairs.append((employees[i], employees[i+1]))
                    
                    # Add strategic pairs within location
                    if len(employees) > 2:
                        # Compare highest with third highest
                        same_level_pairs.append((employees[0], employees[2]))
                        
                        # Compare every third employee if enough employees exist
                        for i in range(0, len(employees)-2, 2):
                            same_level_pairs.append((employees[i], employees[i+2]))
            
            # After location-based comparisons, add cross-location comparisons
            locations = list(job_level_df['Location'].unique())
            if len(locations) > 1:
                for i in range(len(locations)-1):
                    loc1_emp = job_level_df[job_level_df['Location'] == locations[i]].sort_values('Experience', ascending=False).to_dict('records')
                    loc2_emp = job_level_df[job_level_df['Location'] == locations[i+1]].sort_values('Experience', ascending=False).to_dict('records')
                    
                    if loc1_emp and loc2_emp:
                        # Compare top performers from each location
                        same_level_pairs.append((loc1_emp[0], loc2_emp[0]))
                        # Compare second-best if available
                        if len(loc1_emp) > 1 and len(loc2_emp) > 1:
                            same_level_pairs.append((loc1_emp[1], loc2_emp[1]))
        
        # Remove any duplicate pairs or self-comparisons
        unique_pairs = []
        seen_pairs = set()
        
        for pair in same_level_pairs:
            emp1, emp2 = pair[0]['Employee'], pair[1]['Employee']
            if emp1 != emp2:  # Avoid self-comparisons
                pair_key = tuple(sorted([emp1, emp2]))
                if pair_key not in seen_pairs:  # Avoid duplicates
                    seen_pairs.add(pair_key)
                    unique_pairs.append(pair)
        
        self.current_pairs = unique_pairs
        self.current_pair_index = 0
        
        print("\nInitial Comparisons to be made:")
        for pair in unique_pairs:
            print(f"- {pair[0]['Employee']} ({pair[0]['Location']}) vs "
                  f"{pair[1]['Employee']} ({pair[1]['Location']}) "
                  f"[{pair[0]['Job_Level']}, Exp: {pair[0]['Experience']} vs {pair[1]['Experience']}]")

    def get_rankings(self):
        """Return the current rankings for all job levels"""
        final_rankings = {}
        for job_level in self.rankings:
            sorted_rankings = sorted(
                self.rankings[job_level].items(),
                key=lambda x: x[1],
                reverse=True
            )
            final_rankings[job_level] = sorted_rankings
        return final_rankings

ranking_system = EmployeeRanking(df)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form.get('location')
        min_experience = request.form.get('min_experience')
        location = None if location.lower() in ['all', ''] else location
        min_experience = int(min_experience) if min_experience.strip() and min_experience != '0' else None
        
        # Filter employees and prepare clustering
        filtered_df = ranking_system.filter_employees(location=location, min_experience=min_experience)
        ranking_system.prepare_clustering(filtered_df)
        
        return redirect(url_for('compare'))
    return render_template('index.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        winner_id = request.form.get('winner')
        loser_id = request.form.get('loser')
        job_level = request.form.get('job_level')
        
        if winner_id and loser_id:
            # Record the comparison
            ranking_system.record_comparison(winner_id, loser_id, job_level)
    
    # Get next pair for comparison
    pair = ranking_system.get_next_pair()
    
    # If no more valid pairs or we hit a duplicate, redirect to rankings
    if pair is None or (
        pair[0]['Employee'] == pair[1]['Employee'] or  # Same employee comparison
        tuple(sorted([pair[0]['Employee'], pair[1]['Employee']])) in ranking_system.completed_comparisons  # Duplicate comparison
    ):
        print("Comparisons complete, redirecting to rankings...")
        return redirect(url_for('show_rankings'))
    
    return render_template('comparison.html', 
                         employee1=pair[0], 
                         employee2=pair[1])

@app.route('/rankings')
def show_rankings():
    rankings = ranking_system.get_rankings()
    print("Rankings:", rankings)  # Debug print
    if not rankings:
        return "No rankings available yet.", 404
    return render_template('rankings.html', rankings=rankings)

if __name__ == '__main__':
    print("Starting Flask server...")
    # Open browser automatically
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)