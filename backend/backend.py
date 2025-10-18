"""
Flask Backend API for NUSMods Timetable Optimizer
Run with: python backend.py
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from ortools.sat.python import cp_model
from typing import List, Dict, Tuple
from collections import defaultdict
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend


# ==================== NUSMods API Handler ====================

class NUSModsAPI:
    """Fetches module data from NUSMods API"""
    
    BASE_URL = "https://api.nusmods.com/v2"
    ACADEMIC_YEAR = "2024-2025"
    
    @staticmethod
    def get_module_data(module_code: str) -> Dict:
        """Fetch module information including all lesson slots"""
        url = f"{NUSModsAPI.BASE_URL}/{NUSModsAPI.ACADEMIC_YEAR}/modules/{module_code}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {module_code}: {e}")
            return None
    
    @staticmethod
    def parse_lessons(module_data: Dict) -> Dict[str, List[Dict]]:
        """Parse lessons into structured format by lesson type"""
        lessons_by_type = defaultdict(list)
        
        if not module_data or 'semesterData' not in module_data:
            return lessons_by_type
        
        # Get first semester data (Semester 1)
        semester_data = None
        for sem in module_data['semesterData']:
            if sem['semester'] == 1:
                semester_data = sem
                break
        
        if not semester_data or 'timetable' not in semester_data:
            return lessons_by_type
        
        for lesson in semester_data['timetable']:
            lesson_type = lesson['lessonType']
            lessons_by_type[lesson_type].append({
                'classNo': lesson['classNo'],
                'day': lesson['day'],
                'startTime': lesson['startTime'],
                'endTime': lesson['endTime'],
                'venue': lesson['venue'],
                'weeks': lesson.get('weeks')
            })
        
        return dict(lessons_by_type)


# ==================== Timetable Optimizer ====================

class TimetableOptimizer:
    """Optimization engine using Google OR-Tools CP-SAT solver"""
    
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    def __init__(self, modules: List[str], preferences: Dict):
        self.modules = modules
        self.preferences = preferences
        self.module_data = {}
        self.lessons_by_module = {}
        
    def fetch_module_data(self):
        """Fetch all module data from API"""
        print(f"Fetching module data for: {', '.join(self.modules)}")
        for module_code in self.modules:
            data = NUSModsAPI.get_module_data(module_code)
            if data:
                self.module_data[module_code] = data
                self.lessons_by_module[module_code] = NUSModsAPI.parse_lessons(data)
                print(f"✓ Loaded {module_code}: {data.get('title', 'Unknown')}")
            else:
                print(f"✗ Failed to load {module_code}")
    
    def _get_exam_interval_for_module(self, module_code: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parse exam start datetime and compute exam end datetime for a module (if available).
        Returns (start_dt, end_dt) or None if no valid exam info.
        """
        data = self.module_data.get(module_code)
        if not data:
            data = NUSModsAPI.get_module_data(module_code)
            if data:
                self.module_data[module_code] = data
            else:
                print(f"  No data found for {module_code}")
                return None

        sem_data = data.get('semesterData', [])
        print(f"  {module_code}: Found {len(sem_data)} semester(s)")
        
        # Check ALL semesters, not just semester 1
        for s in sem_data:
            exam_date = s.get('examDate')
            exam_duration = s.get('examDuration')
            semester = s.get('semester')
            print(f"    Sem {semester}: examDate={exam_date}, duration={exam_duration}")
            
            if exam_date and exam_duration:
                try:
                    start_dt = datetime.fromisoformat(exam_date)
                except Exception:
                    try:
                        start_dt = datetime.fromisoformat(exam_date.replace('Z', '+00:00'))
                    except Exception:
                        print(f"    Failed to parse exam date: {exam_date}")
                        continue
                try:
                    duration_min = int(exam_duration)
                except Exception:
                    duration_min = 120
                end_dt = start_dt + timedelta(minutes=duration_min)
                print(f"    ✓ {module_code} exam: {start_dt} to {end_dt}")
                return (start_dt, end_dt)
        
        print(f"    No valid exam info for {module_code}")
        return None

    def find_exam_clashes(self, module_codes: List[str]) -> List[Dict]:
        """
        Returns list of exam clash groups:
        { 'modules': [..], 'overlap_start': ISO, 'overlap_end': ISO }
        """
        print(f"Checking exam clashes for: {module_codes}")
        intervals = []
        for mod in module_codes:
            interval = self._get_exam_interval_for_module(mod)
            if interval is not None:
                intervals.append({'module': mod, 'start': interval[0], 'end': interval[1]})

        print(f"Found {len(intervals)} modules with exam times")
        if len(intervals) < 2:
            print("Not enough modules with exam times to check for clashes")
            return []

        intervals.sort(key=lambda x: x['start'])

        groups: List[List[Dict]] = []
        current_group = [intervals[0]]
        current_end = intervals[0]['end']

        for it in intervals[1:]:
            if it['start'] < current_end:
                current_group.append(it)
                if it['end'] > current_end:
                    current_end = it['end']
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [it]
                current_end = it['end']

        if len(current_group) > 1:
            groups.append(current_group)

        result = []
        for g in groups:
            overlap_start = min(item['start'] for item in g)
            overlap_end = max(item['end'] for item in g)
            result.append({
                'modules': [item['module'] for item in g],
                'overlap_start': overlap_start.isoformat(),
                'overlap_end': overlap_end.isoformat(),
            })
        return result
    
    @staticmethod
    def time_to_slot(time_str: str) -> int:
        """Convert time string (e.g., '0900') to slot index (30-min intervals from 8am)"""
        hours = int(time_str[:2])
        minutes = int(time_str[2:])
        return (hours - 8) * 2 + (minutes // 30)
    
    @staticmethod
    def slot_to_time(slot: int) -> str:
        """Convert slot index to time string"""
        hours = 8 + (slot // 2)
        minutes = (slot % 2) * 30
        return f"{hours:02d}{minutes:02d}"
    
    def optimize(self) -> List[Dict]:
        """Run optimization and return multiple timetable solutions"""
        self.fetch_module_data()
        
        # Check if we have valid modules
        if not self.lessons_by_module:
            return []
        
        # Check if all modules have lessons
        for module, lessons in self.lessons_by_module.items():
            if not lessons:
                print(f"Warning: {module} has no lessons available")
        
        model = cp_model.CpModel()
        
        # Decision variables: for each module and lesson type, pick one class
        class_vars = {}
        for module, lessons_by_type in self.lessons_by_module.items():
            if not lessons_by_type:
                continue
                
            class_vars[module] = {}
            for lesson_type, classes in lessons_by_type.items():
                class_vars[module][lesson_type] = {}
                for lesson in classes:
                    class_id = lesson['classNo']
                    var = model.NewBoolVar(f"{module}_{lesson_type}_{class_id}")
                    class_vars[module][lesson_type][class_id] = (var, lesson)
                
                # Constraint: exactly one class must be chosen per lesson type
                model.Add(sum(var for var, _ in class_vars[module][lesson_type].values()) == 1)
        
        if not class_vars:
            print("No valid class variables created")
            return []
        
        # Constraint: No overlapping classes
        self._add_no_overlap_constraints(model, class_vars)
        
        # Soft constraints (preferences) as objectives
        objective_terms = []
        
        if self.preferences.get('noMorningClasses'):
            morning_penalty = self._add_no_morning_classes(model, class_vars)
            if morning_penalty is not None:
                objective_terms.append(morning_penalty)
        
        if self.preferences.get('freeFridays'):
            friday_penalty = self._add_free_fridays(model, class_vars)
            if friday_penalty is not None:
                objective_terms.append(friday_penalty)
        
        if self.preferences.get('lunchBreak'):
            lunch_penalty = self._add_lunch_break_constraint(model, class_vars)
            if lunch_penalty is not None:
                objective_terms.append(lunch_penalty)
        
        if self.preferences.get('minimizeTravel'):
            travel_penalty = self._add_minimize_travel(model, class_vars)
            if travel_penalty is not None:
                objective_terms.append(travel_penalty)
        
        if self.preferences.get('compactSchedule'):
            gap_penalty = self._add_compact_schedule(model, class_vars)
            if gap_penalty is not None:
                objective_terms.append(gap_penalty)
        
        # Set objective: minimize penalties
        if objective_terms:
            model.Minimize(sum(objective_terms))
        
        # Solve and collect multiple solutions
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.enumerate_all_solutions = True  # Key: enumerate all solutions
        
        solutions = []
        
        class SolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, variables):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._variables = variables
                self.solutions = []
            
            def on_solution_callback(self):
                solution = {}
                for module, types in self._variables.items():
                    solution[module] = {}
                    for lesson_type, classes in types.items():
                        for class_id, (var, lesson) in classes.items():
                            if self.Value(var):
                                solution[module][lesson_type] = lesson
                                break
                self.solutions.append(solution)
        
        solution_collector = SolutionCollector(class_vars)
        status = solver.Solve(model, solution_collector)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"Found {len(solution_collector.solutions)} solution(s)")
            return self._format_solutions(solution_collector.solutions)
        else:
            print(f"No feasible solution found. Status: {status}")
            return []
    
    def _add_no_overlap_constraints(self, model, class_vars):
        """Ensure no two classes overlap in time"""
        all_classes = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    all_classes.append((var, lesson))
        
        for i in range(len(all_classes)):
            for j in range(i + 1, len(all_classes)):
                var1, lesson1 = all_classes[i]
                var2, lesson2 = all_classes[j]
                
                if self._lessons_overlap(lesson1, lesson2):
                    # If both are selected, they overlap - not allowed
                    model.Add(var1 + var2 <= 1)
    
    def _lessons_overlap(self, lesson1: Dict, lesson2: Dict) -> bool:
        """Check if two lessons overlap in time and weeks (if weeks info exists)."""

        # Different day => no overlap
        if lesson1.get('day') != lesson2.get('day'):
            return False

        start1 = self.time_to_slot(lesson1['startTime'])
        end1 = self.time_to_slot(lesson1['endTime'])
        start2 = self.time_to_slot(lesson2['startTime'])
        end2 = self.time_to_slot(lesson2['endTime'])

        time_overlap = not (end1 <= start2 or end2 <= start1)
        if not time_overlap:
            return False

        # If both lessons include explicit weeks info, check week intersection.
        weeks1 = lesson1.get('weeks')  # expected to be a list like [1,2,3] if parse provides it
        weeks2 = lesson2.get('weeks')

        if weeks1 is None or weeks2 is None:
            # Fallback behavior: no weeks info — treat as overlapping based on time only.
            # If you prefer conservative behavior (treat unknown weeks as conflict), change to `return True` here.
            return True

        try:
            set1 = set(int(w) for w in weeks1)
            set2 = set(int(w) for w in weeks2)
        except Exception:
            # If parsing fails, be conservative and treat as overlapping
            return True

        # Return true only if weeks intersect
        return len(set1.intersection(set2)) > 0
    
    def _add_no_morning_classes(self, model, class_vars):
        """Penalize classes before 10am"""
        morning_vars = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    start_slot = self.time_to_slot(lesson['startTime'])
                    if start_slot < 4:  # Before 10am (8am = 0, 10am = 4)
                        morning_vars.append(var)
        
        if morning_vars:
            penalty = model.NewIntVar(0, len(morning_vars) * 10, 'morning_penalty')
            model.Add(penalty == sum(morning_vars) * 10)
            return penalty
        return None
    
    def _add_free_fridays(self, model, class_vars):
        """Penalize Friday classes"""
        friday_vars = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    if lesson['day'] == 'Friday':
                        friday_vars.append(var)
        
        if friday_vars:
            penalty = model.NewIntVar(0, len(friday_vars) * 15, 'friday_penalty')
            model.Add(penalty == sum(friday_vars) * 15)
            return penalty
        return None
    
    def _add_lunch_break_constraint(self, model, class_vars):
        """Penalize classes that overlap with lunch time (12pm-2pm)"""
        lunch_overlap_vars = []
        
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    start_slot = self.time_to_slot(lesson['startTime'])
                    end_slot = self.time_to_slot(lesson['endTime'])
                    
                    # Lunch time: 12pm = slot 8, 2pm = slot 12
                    # Check if class overlaps with lunch (not completely before or after)
                    if not (end_slot <= 8 or start_slot >= 12):
                        lunch_overlap_vars.append(var)
        
        if lunch_overlap_vars:
            # Lower penalty (8) so it prefers lunch breaks but doesn't make solution infeasible
            penalty = model.NewIntVar(0, len(lunch_overlap_vars) * 8, 'lunch_penalty')
            model.Add(penalty == sum(lunch_overlap_vars) * 8)
            return penalty
        return None
    
    def _add_minimize_travel(self, model, class_vars):
        """Penalize classes that require traveling between distant locations on the same day"""
        # Define location groups (buildings that are close together)
        location_groups = {
            'COM': ['COM1', 'COM2', 'COM3', 'I3'],  # Computing buildings
            'LT': ['LT', 'AS'],  # Lecture theaters in Arts/Science
            'S': ['S17', 'S16', 'S14', 'S11'],  # Science buildings
            'E': ['E1', 'E2', 'E3', 'E4', 'E5', 'EA', 'EW'],  # Engineering
            'UTOWN': ['UT', 'RC4'],  # UTown area
        }
        
        def get_location_group(venue):
            """Get the location group for a venue"""
            if not venue:
                return 'UNKNOWN'
            # Extract building prefix (e.g., 'COM1' from 'COM1-B111')
            for group, buildings in location_groups.items():
                for building in buildings:
                    if venue.startswith(building):
                        return group
            return 'OTHER'
        
        # Group classes by day and check for travel between different areas
        travel_penalty_vars = []
        
        # Get all possible class combinations
        all_class_options = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    all_class_options.append((var, lesson))
        
        # Check pairs of classes on the same day
        for i in range(len(all_class_options)):
            for j in range(i + 1, len(all_class_options)):
                var1, lesson1 = all_class_options[i]
                var2, lesson2 = all_class_options[j]
                
                # Only check if they're on the same day
                if lesson1['day'] == lesson2['day']:
                    loc1 = get_location_group(lesson1['venue'])
                    loc2 = get_location_group(lesson2['venue'])
                    
                    # If they're in different location groups, add penalty
                    if loc1 != loc2 and loc1 != 'OTHER' and loc2 != 'OTHER':
                        # Create a variable that's 1 if both classes are selected
                        travel_var = model.NewBoolVar(f'travel_{i}_{j}')
                        # travel_var is 1 only if both var1 and var2 are selected
                        model.Add(var1 + var2 == 2).OnlyEnforceIf(travel_var)
                        model.Add(var1 + var2 < 2).OnlyEnforceIf(travel_var.Not())
                        travel_penalty_vars.append(travel_var)
        
        if travel_penalty_vars:
            penalty = model.NewIntVar(0, len(travel_penalty_vars) * 5, 'travel_penalty')
            model.Add(penalty == sum(travel_penalty_vars) * 5)
            return penalty
        return None
    
    def _add_compact_schedule(self, model, class_vars):
        """Penalize large gaps between classes on the same day"""
        gap_penalty_vars = []
        
        # Get all possible class combinations
        all_class_options = []
        for module in class_vars:
            for lesson_type in class_vars[module]:
                for class_id, (var, lesson) in class_vars[module][lesson_type].items():
                    all_class_options.append((var, lesson))
        
        # Check pairs of classes on the same day for gaps
        for i in range(len(all_class_options)):
            for j in range(i + 1, len(all_class_options)):
                var1, lesson1 = all_class_options[i]
                var2, lesson2 = all_class_options[j]
                
                # Only check if they're on the same day
                if lesson1['day'] == lesson2['day']:
                    # Calculate gap between classes
                    end1 = self.time_to_slot(lesson1['endTime'])
                    start2 = self.time_to_slot(lesson2['startTime'])
                    
                    # Ensure lesson1 ends before lesson2 starts
                    if end1 <= start2:
                        gap_slots = start2 - end1
                        # Penalize gaps > 2 hours (4 slots = 2 hours)
                        if gap_slots > 4:
                            # Create variable for when both classes are selected
                            gap_var = model.NewBoolVar(f'gap_{i}_{j}')
                            model.Add(var1 + var2 == 2).OnlyEnforceIf(gap_var)
                            model.Add(var1 + var2 < 2).OnlyEnforceIf(gap_var.Not())
                            # Penalty increases with gap size
                            gap_penalty_vars.extend([gap_var] * (gap_slots - 4))
        
        if gap_penalty_vars:
            penalty = model.NewIntVar(0, len(gap_penalty_vars) * 2, 'gap_penalty')
            model.Add(penalty == sum(gap_penalty_vars) * 2)
            return penalty
        return None
    
    def _format_solutions(self, raw_solutions: List[Dict]) -> List[Dict]:
        """Format solutions into readable timetables"""
        formatted = []
        for idx, solution in enumerate(raw_solutions):
            schedule = []
            for module, lessons in solution.items():
                for lesson_type, lesson_data in lessons.items():
                    schedule.append({
                        'module': module,
                        'type': lesson_type,
                        'classNo': lesson_data['classNo'],
                        'day': lesson_data['day'],
                        'startTime': lesson_data['startTime'],
                        'endTime': lesson_data['endTime'],
                        'venue': lesson_data['venue'],
                        'weeks': lesson_data.get('weeks')
                    })
            
            # Sort by day and time
            schedule.sort(key=lambda x: (self.DAYS.index(x['day']), x['startTime']))
            
            formatted.append({
                'id': idx + 1,
                'schedule': schedule,
                'score': self._calculate_score(schedule),
                'tradeoffs': self._generate_tradeoffs(schedule)
            })
        
        return formatted
    
    def _calculate_score(self, schedule: List[Dict]) -> int:
        """Calculate timetable quality score"""
        score = 100
        
        # Deduct for morning classes
        if self.preferences.get('noMorningClasses'):
            morning_count = sum(1 for s in schedule if self.time_to_slot(s['startTime']) < 4)
            score -= morning_count * 5
            print(f"  Morning classes: {morning_count}, Score after: {score}")
        
        # Deduct for Friday classes
        if self.preferences.get('freeFridays'):
            friday_count = sum(1 for s in schedule if s['day'] == 'Friday')
            score -= friday_count * 10
            print(f"  Friday classes: {friday_count}, Score after: {score}")
        
        # Deduct for lunch break conflicts
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            conflict_count = len(lunch_conflicts)
            score -= conflict_count * 8  # 8 points per day with lunch conflict
            print(f"  Lunch conflicts on {conflict_count} day(s): {lunch_conflicts}, Score after: {score}")
        
        # Deduct for campus travel
        if self.preferences.get('minimizeTravel'):
            travel_count = self._count_campus_travel(schedule)
            score -= travel_count * 3  # 3 points per cross-campus trip
            print(f"  Cross-campus trips: {travel_count}, Score after: {score}")
        
        # Deduct for gaps in schedule
        if self.preferences.get('compactSchedule'):
            gap_hours = self._calculate_total_gaps(schedule)
            score -= gap_hours * 2  # 2 points per hour of gaps
            print(f"  Total gap hours: {gap_hours}, Score after: {score}")
        
        print(f"  Final score: {score}")
        return max(0, score)
    
    def _generate_tradeoffs(self, schedule: List[Dict]) -> List[str]:
        """Generate human-readable tradeoff explanations"""
        tradeoffs = []
        
        # Check for morning classes
        morning_classes = [s for s in schedule if self.time_to_slot(s['startTime']) < 4]
        if self.preferences.get('noMorningClasses'):
            if not morning_classes:
                tradeoffs.append('✓ No classes before 10am')
            else:
                tradeoffs.append(f'⚠ {len(morning_classes)} morning class(es) scheduled (before 10am)')
        
        # Check for Friday classes
        friday_classes = [s for s in schedule if s['day'] == 'Friday']
        if self.preferences.get('freeFridays'):
            if not friday_classes:
                tradeoffs.append('✓ Fridays are completely free')
            else:
                tradeoffs.append(f'⚠ {len(friday_classes)} class(es) on Friday')
        
        # Check for lunch breaks - detailed by day
        if self.preferences.get('lunchBreak'):
            lunch_conflicts = self._get_lunch_conflicts(schedule)
            if not lunch_conflicts:
                tradeoffs.append('✓ Lunch breaks (12-2pm) preserved on all days')
            else:
                days_str = ', '.join(lunch_conflicts)
                tradeoffs.append(f'⚠ Lunch break conflicts on: {days_str}')
        
        # Compact schedule analysis
        days_with_classes = set(s['day'] for s in schedule)
        tradeoffs.append(f'📅 Classes spread across {len(days_with_classes)} day(s)')
        
        # Show gap information if compact schedule preferred
        if self.preferences.get('compactSchedule'):
            total_gaps = self._calculate_total_gaps(schedule)
            if total_gaps == 0:
                tradeoffs.append('✓ No gaps between classes (fully compact)')
            else:
                tradeoffs.append(f'⏰ {total_gaps} hour(s) of gaps between classes')
        
        return tradeoffs
    
    def _get_lunch_conflicts(self, schedule: List[Dict]) -> List[str]:
        """Get list of days with lunch time conflicts"""
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        conflict_days = []
        for day, slots in days_schedule.items():
            for slot in slots:
                start = self.time_to_slot(slot['startTime'])
                end = self.time_to_slot(slot['endTime'])
                # Check if class overlaps with lunch (12pm = slot 8, 2pm = slot 12)
                if not (end <= 8 or start >= 12):
                    conflict_days.append(day)
                    break  # Only count each day once
        
        return conflict_days
    
    def _count_campus_travel(self, schedule: List[Dict]) -> int:
        """Count number of cross-campus trips in the schedule"""
        location_groups = {
            'COM': ['COM1', 'COM2', 'COM3', 'I3'],
            'LT': ['LT', 'AS'],
            'S': ['S17', 'S16', 'S14', 'S11'],
            'E': ['E1', 'E2', 'E3', 'E4', 'E5', 'EA', 'EW'],
            'UTOWN': ['UT', 'RC4'],
        }
        
        def get_location_group(venue):
            if not venue:
                return 'UNKNOWN'
            for group, buildings in location_groups.items():
                for building in buildings:
                    if venue.startswith(building):
                        return group
            return 'OTHER'
        
        # Group by day
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        travel_count = 0
        for day, slots in days_schedule.items():
            # Sort by time
            sorted_slots = sorted(slots, key=lambda x: x['startTime'])
            # Count transitions between different location groups
            for i in range(len(sorted_slots) - 1):
                loc1 = get_location_group(sorted_slots[i]['venue'])
                loc2 = get_location_group(sorted_slots[i + 1]['venue'])
                if loc1 != loc2 and loc1 != 'OTHER' and loc2 != 'OTHER':
                    travel_count += 1
        
        return travel_count
    
    def _calculate_total_gaps(self, schedule: List[Dict]) -> int:
        """Calculate total hours of gaps between classes"""
        # Group by day
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        total_gap_hours = 0
        for day, slots in days_schedule.items():
            # Sort by start time
            sorted_slots = sorted(slots, key=lambda x: x['startTime'])
            
            # Calculate gaps between consecutive classes
            for i in range(len(sorted_slots) - 1):
                end_time = self.time_to_slot(sorted_slots[i]['endTime'])
                start_time = self.time_to_slot(sorted_slots[i + 1]['startTime'])
                
                gap_slots = start_time - end_time
                # Only count gaps > 0 (30 min intervals)
                if gap_slots > 0:
                    # Convert slots to hours (2 slots = 1 hour)
                    gap_hours = gap_slots // 2
                    total_gap_hours += gap_hours
        
        return total_gap_hours
    
    def _check_lunch_break(self, schedule: List[Dict]) -> bool:
        """Check if there's a lunch break from 12-2pm on all days"""
        days_schedule = defaultdict(list)
        for slot in schedule:
            days_schedule[slot['day']].append(slot)
        
        for day, slots in days_schedule.items():
            for slot in slots:
                start = self.time_to_slot(slot['startTime'])
                end = self.time_to_slot(slot['endTime'])
                # Check if class overlaps with lunch (12pm = slot 8, 2pm = slot 12)
                if not (end <= 8 or start >= 12):
                    return False
        return True


# ==================== Flask API Routes ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})


@app.route('/api/optimize', methods=['POST'])
def optimize_timetable():
    """Main optimization endpoint"""
    try:
        data = request.json
        modules = data.get('modules', [])
        preferences = data.get('preferences', {})

        print(f"\n{'='*60}")
        print(f"Received optimization request:")
        print(f"Modules: {modules}")
        print(f"Preferences: {preferences}")
        print(f"{'='*60}\n")

        if not modules:
            return jsonify({'error': 'No modules provided'}), 400

        optimizer = TimetableOptimizer(modules, preferences)
        # Ensure module_data and lessons_by_module are populated for exam checking
        optimizer.fetch_module_data()

        exam_clashes = optimizer.find_exam_clashes(modules)
        print(f"Exam clash detection result: {len(exam_clashes)} clashes found")
        
        # Always block if exam clashes are detected (minimal change: remove preference check)
        if exam_clashes:
            return jsonify({
                'error': 'Exam clashes detected between your selected modules',
                'examClashes': exam_clashes,
                'count': len(exam_clashes),
            }), 409

        timetables = optimizer.optimize()

        if not timetables:
            return jsonify({
                'error': 'No feasible timetables found. This might be due to:\n'
                         '- Module codes not available for current semester\n'
                         '- Too many conflicting constraints\n'
                         '- Invalid module codes',
                'examClashes': exam_clashes,
            }), 404

        return jsonify({
            'success': True,
            'timetables': timetables,
            'count': len(timetables),
            'examClashes': exam_clashes,
        })
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/modules/<module_code>', methods=['GET'])
def get_module_info(module_code):
    """Get information about a specific module"""
    try:
        data = NUSModsAPI.get_module_data(module_code)
        if not data:
            return jsonify({'error': f'Module {module_code} not found'}), 404
        
        lessons = NUSModsAPI.parse_lessons(data)
        
        return jsonify({
            'code': module_code,
            'title': data.get('title'),
            'description': data.get('description'),
            'lessons': lessons
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NUSMods Timetable Optimizer Backend")
    print("="*60)
    print("Server starting on http://localhost:5000")
    print("API Endpoints:")
    print("  - GET  /api/health")
    print("  - POST /api/optimize")
    print("  - GET  /api/modules/<module_code>")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)