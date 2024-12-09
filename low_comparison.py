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
data = {'Employee': ['Jijo', 'Jobin', 'Suhair', 'Anakha', 'Ponny', 'Aneesh R', 'Colin', 'Yoosef', 'Shinoj', 'Arjun', 'Najmal', 'Sreelakshmi','Anand', 'Raghu'],
        'Job_Level': ['Senior', 'Mid-Level', 'Senior', 'Junior', 'Mid-Level', 'Senior', 'Junior', 'Mid-Level', 'Senior', 'Junior', 'Junior', 'Mid-Level', 'Mid-Level', 'Senior'],
        'Experience': [17, 7, 16, 1, 5, 10, 2, 9, 14, 3, 2, 6, 7, 14],
        'Location': ['Utah', 'Kochi', 'Utah', 'Kochi', 'Kochi', 'Utah', 'Kochi', 'Kochi', 'Kochi', ' Kochi', 'Kochi', 'Kochi', 'Kochi', 'Utah']}
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
            emp1_name, emp2_name = emp1['Employee'], emp2['Employee']
            
            print(f"Considering pair: {emp1_name} vs {emp2_name}")
            
            # Skip if this comparison has already been done
            comparison_key = tuple(sorted([emp1_name, emp2_name]))
            if comparison_key in self.completed_comparisons:
                print(f"Skipping {emp1_name} vs {emp2_name}: Already compared")
                self.current_pair_index += 1
                continue
            
            # Check for direct win/loss
            if emp1_name in self.wins[emp2_name]:
                print(f"Skipping {emp1_name} vs {emp2_name}: {emp2_name} already beat {emp1_name}")
                self.current_pair_index += 1
                continue
            if emp2_name in self.wins[emp1_name]:
                print(f"Skipping {emp1_name} vs {emp2_name}: {emp1_name} already beat {emp2_name}")
                self.current_pair_index += 1
                continue
            
            # Check for transitive wins/losses
            if any(emp2_name in self.losses[loser] for loser in self.wins[emp1_name]):
                print(f"Skipping {emp1_name} vs {emp2_name}: {emp1_name} wins transitively")
                self.record_comparison(emp1_name, emp2_name, emp1['Job_Level'])
                self.current_pair_index += 1
                continue
            
            if any(emp1_name in self.losses[loser] for loser in self.wins[emp2_name]):
                print(f"Skipping {emp1_name} vs {emp2_name}: {emp2_name} wins transitively")
                self.record_comparison(emp2_name, emp1_name, emp1['Job_Level'])
                self.current_pair_index += 1
                continue
            
            self.current_pair_index += 1
            return pair
        
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

    def filter_employees(self, job_level=None, location=None, min_experience=None):
        filtered_df = self.df.copy()
        if job_level:
            filtered_df = filtered_df[filtered_df['Job_Level'] == job_level]
        if location:
            filtered_df = filtered_df[filtered_df['Location'] == location]
        if min_experience is not None:
            filtered_df = filtered_df[filtered_df['Experience'] >= min_experience]
        return filtered_df

    def prepare_clustering(self, filtered_df):
        # Reset all data before starting new comparisons
        self.reset()
        
        # Initialize tracking dictionaries
        self.wins = {emp['Employee']: set() for emp in filtered_df.to_dict('records')}
        self.losses = {emp['Employee']: set() for emp in filtered_df.to_dict('records')}
        
        # Create pairs between employees in the same job level
        same_level_pairs = []
        for job_level in filtered_df['Job_Level'].unique():
            # Get employees in the same job level
            same_level_df = filtered_df[filtered_df['Job_Level'] == job_level]
            
            if len(same_level_df) > 1:
                employees = same_level_df.to_dict('records')
                # Sort by experience to compare similar experiences first
                employees.sort(key=lambda x: x['Experience'], reverse=True)
                
                # Create initial pairs between adjacent employees
                for i in range(len(employees)-1):
                    same_level_pairs.append((employees[i], employees[i+1]))
                
                # Add some additional strategic pairs
                # Compare first with third if more than 2 employees
                if len(employees) > 2:
                    same_level_pairs.append((employees[0], employees[2]))
                
                # Compare middle employees if more than 3
                if len(employees) > 3:
                    mid = len(employees) // 2
                    same_level_pairs.append((employees[mid-1], employees[mid+1]))
        
        # Store pairs
        self.current_pairs = same_level_pairs
        self.current_pair_index = 0
        
        print("\nInitial Comparisons to be made:")
        for pair in same_level_pairs:
            print(f"- {pair[0]['Employee']} vs {pair[1]['Employee']} ({pair[0]['Job_Level']})")

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
            ranking_system.record_comparison(winner_id, loser_id, job_level)
    
    # Get next pair for comparison
    pair = ranking_system.get_next_pair()
    
    if pair is None:
        # No more pairs to compare, redirect to rankings
        print("No more pairs, redirecting to rankings...")  # Debug print
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