import json
import os
import streamlit as st
import pandas as pd
import random


@st.cache_data(ttl=60)
def load_data() -> dict:
	data_dir = os.path.join(os.path.dirname(__file__), 'data')
	json_path = os.path.join(data_dir, 'all_weeks_assignments.json')
	csv_path = os.path.join(data_dir, 'All_Weeks_Questions.csv')

	if os.path.exists(json_path):
		with open(json_path, 'r', encoding='utf-8') as f:
			raw = json.load(f)
			# Support two schemas:
			# 1) {'weeks': {'Week 1': [questions], ...}}
			# 2) {'course': ..., 'assignments': [{ 'week': 1, 'questions': [...]}, ...]}
			if isinstance(raw, dict) and 'weeks' in raw:
				return raw
			weeks: dict = {}
			assignments = raw.get('assignments', []) if isinstance(raw, dict) else []
			for a in assignments:
				week_num = a.get('week')
				week_key = f"Week {week_num}" if week_num is not None else str(a.get('title', 'Week'))
				qs = []
				for q in a.get('questions', []):
					q_type = q.get('question_type', 'MCQ')
					qs.append({
						'question_text': q.get('question_text', ''),
						'options': q.get('options', []),
						'question_type': q_type,
						'correct_answers': q.get('correct_answers') if q_type == 'MSQ' else None,
						'correct_answer': q.get('correct_answer'),
						'points': q.get('points', 1),
					})
				weeks.setdefault(week_key, []).extend(qs)
			return {'weeks': weeks}
	elif os.path.exists(csv_path):
		df = pd.read_csv(csv_path)
		weeks = {}
		for _, row in df.iterrows():
			week = str(row['Week']).strip()
			question_text = str(row['Question']).strip()
			options = [opt.strip() for opt in str(row['Options']).split(';')]
			correct = str(row['Correct Option(s)']).split(';') if ';' in str(row['Correct Option(s)']) else [str(row['Correct Option(s)'])]
			q_type = 'MSQ' if len(correct) > 1 else 'MCQ'
			if week not in weeks:
				weeks[week] = []
			weeks[week].append({
				'question_text': question_text,
				'options': options,
				'question_type': q_type,
				'correct_answers': [c.strip() for c in correct] if q_type == 'MSQ' else None,
				'correct_answer': correct[0].strip() if q_type == 'MCQ' else None,
				'points': 1,
			})
		return {'weeks': weeks}
	else:
		return {'weeks': {}}



def calculate_score(questions: list, user_answers: dict):
	correct_answers = 0
	earned_points = 0
	results = []
	total_possible_points = sum(q.get('points', 1) for q in questions)

	for i, question in enumerate(questions):
		key = f"q{i+1}"
		user_answer = user_answers.get(key)

		qtype = question.get('question_type', 'MCQ')
		if qtype == 'MCQ':
			correct_answer = question.get('correct_answer', '')
			is_correct = user_answer == correct_answer
			if is_correct:
				correct_answers += 1
				earned_points += question.get('points', 1)
			results.append({
				'question_num': i + 1,
				'question_text': question['question_text'],
				'user_answer': user_answer,
				'correct_answer': correct_answer,
				'is_correct': is_correct,
				'points': question.get('points', 1)
			})
		elif qtype == 'MSQ':
			correct_list = question.get('correct_answers', [])
			if not correct_list and 'correct_answer' in question:
				correct_list = [question['correct_answer']]
			selected = user_answer if isinstance(user_answer, list) else []
			all_correct = all(opt in selected for opt in correct_list)
			no_incorrect = not any(opt in selected for opt in question['options'] if opt not in correct_list)
			is_correct = all_correct and no_incorrect
			if is_correct:
				correct_answers += 1
				earned_points += question.get('points', 1)
			results.append({
				'question_num': i + 1,
				'question_text': question['question_text'],
				'user_answer': selected,
				'correct_answer': correct_list,
				'is_correct': is_correct,
				'points': question.get('points', 1)
			})
		elif qtype == 'NUMERIC':
			correct_numeric = str(question.get('correct_answer', '')).strip()
			user_str = '' if user_answer is None else str(user_answer).strip()
			is_correct = False
			# Try numeric comparison if possible
			try:
				is_correct = float(user_str) == float(correct_numeric)
			except Exception:
				is_correct = user_str == correct_numeric
			if is_correct:
				correct_answers += 1
				earned_points += question.get('points', 1)
			results.append({
				'question_num': i + 1,
				'question_text': question['question_text'],
				'user_answer': user_str,
				'correct_answer': correct_numeric,
				'is_correct': is_correct,
				'points': question.get('points', 1)
			})
		else:
			# Fallback treat as MCQ
			correct_answer = question.get('correct_answer', '')
			is_correct = user_answer == correct_answer
			if is_correct:
				correct_answers += 1
				earned_points += question.get('points', 1)
			results.append({
				'question_num': i + 1,
				'question_text': question['question_text'],
				'user_answer': user_answer,
				'correct_answer': correct_answer,
				'is_correct': is_correct,
				'points': question.get('points', 1)
			})

	percentage = (correct_answers / len(questions)) * 100 if questions else 0
	points_percentage = (earned_points / total_possible_points) * 100 if total_possible_points else 0
	return {
		'total_questions': len(questions),
		'correct_answers': correct_answers,
		'percentage': percentage,
		'total_possible_points': total_possible_points,
		'earned_points': earned_points,
		'points_percentage': points_percentage,
		'results': results,
	}


def display_question(question: dict, question_num: int, total_questions: int):
	# Dedupe options
	if 'options' in question:
		question['options'] = list(dict.fromkeys(question['options']))

	st.markdown(f"**Question {question_num} of {total_questions} ({question.get('points',1)} point{'s' if question.get('points',1) > 1 else ''})**")
	st.markdown(f"**{question['question_text']}**")

	key = f"q{question_num}"
	qtype = question.get('question_type', 'MCQ')
	if qtype == 'MCQ':
		choice = st.radio("Select your answer:", question.get('options', []), key=key, index=None)
		st.markdown("---")
		return choice
	elif qtype == 'MSQ':
		selected = []
		st.write("**Select all that apply:**")
		for i, opt in enumerate(question.get('options', [])):
			if st.checkbox(opt, key=f"{key}_option_{i}"):
				selected.append(opt)
		st.markdown("---")
		return selected
	else:
		# NUMERIC or other free-text numeric answer
		val = st.text_input("Enter numeric answer:", key=key)
		st.markdown("---")
		return val.strip()


def main():
	st.set_page_config(page_title="🤖 Introduction to LLM - Practice Assignment Quiz", layout="wide")
	# Grand header styling
	st.markdown(
		"""
		<style>
			.main-header-llm {
				font-size: 2.8rem;
				font-weight: 800;
				text-align: center;
				margin: 0.5rem 0 1rem 0;
				background: linear-gradient(90deg, #1f77b4, #ff7f0e);
				-webkit-background-clip: text;
				background-clip: text;
				color: transparent;
			}
			.sub-header-llm {
				text-align: center;
				color: #4f4f4f;
				font-size: 1rem;
				margin-bottom: 1.2rem;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)
	st.markdown('<div class="main-header-llm">🤖 Introduction to LLM - Practice Assignment Quiz</div>', unsafe_allow_html=True)
	st.markdown('<div class="sub-header-llm">Practice assignments for all 12 weeks • Select weeks • Take the quiz • See results</div>', unsafe_allow_html=True)
	# Credit disclaimer under subtitle
	st.markdown('<div style="text-align:center;color:#6b7280;font-size:0.9rem;margin-top:-0.2rem;margin-bottom:1.0rem;">We do not own any of the content on this website. All credits to NPTEL.</div>', unsafe_allow_html=True)

	data = load_data()
	weeks = list(data.get('weeks', {}).keys())
	if not weeks:
		st.info("Add your CSV or JSON under data/ and refresh.")
		return

	# Sidebar week selection (numeric labels)
	def week_num(w):
		return int(''.join(filter(str.isdigit, w))) if any(c.isdigit() for c in w) else w
	weeks = sorted(weeks, key=lambda w: (week_num(w), w))
	numeric_weeks = [week_num(w) for w in weeks]
	st.sidebar.markdown("## 📚 Select Weeks")
	selected_numbers = st.sidebar.multiselect("Choose weeks:", options=numeric_weeks, default=numeric_weeks[:2] if len(numeric_weeks) >= 2 else numeric_weeks)
	selected_weeks = [w for w in weeks if week_num(w) in selected_numbers]

	# Collect and show week metrics
	week_info = []
	all_questions = []
	for w in selected_weeks:
		qs = data['weeks'][w]
		week_info.append({'week': week_num(w), 'questions': qs})
		for q in qs:
			all_questions.append(q)

	if not all_questions:
		st.warning("No questions found for selected weeks.")
		return

	st.markdown("### 📋 Selected Weeks")
	cols = st.columns(min(6, len(week_info)))
	for i, wk in enumerate(week_info):
		with cols[i % len(cols)]:
			st.metric(label=f"Week {wk['week']}", value=f"{len(wk['questions'])} Q")

	st.markdown(f"**Total Questions:** {len(all_questions)} across {len(selected_weeks)} week{'s' if len(selected_weeks) > 1 else ''}")

	# Options
	st.markdown("### ⚙️ Quiz Options")
	col1, col2 = st.columns(2)
	with col1:
		shuffle_questions = st.checkbox("Shuffle Questions", value=False)
	with col2:
		shuffle_options = st.checkbox("Shuffle Options", value=False, help="Randomize the order of options inside each question")

	# Informational note about shuffle behavior
	st.markdown('<div style="color:#6b7280;font-size:0.9rem;margin-top:0.25rem;">Note: Shuffle applies once when enabled or when weeks change. Interacting with questions will not reshuffle automatically.</div>', unsafe_allow_html=True)

	# Stable one-time shuffle using session state
	if 'shuffle_enabled' not in st.session_state:
		st.session_state.shuffle_enabled = False
	if 'question_order' not in st.session_state:
		st.session_state.question_order = list(range(len(all_questions)))
	if 'shuffle_options_enabled' not in st.session_state:
		st.session_state.shuffle_options_enabled = False
	if 'option_orders' not in st.session_state:
		st.session_state.option_orders = {}

	# Regenerate order when toggling on, or when question set changes length
	if shuffle_questions:
		if (not st.session_state.shuffle_enabled) or (len(st.session_state.question_order) != len(all_questions)):
			order = list(range(len(all_questions)))
			random.shuffle(order)
			st.session_state.question_order = order
			st.session_state.shuffle_enabled = True
	else:
		# Reset to natural order when shuffle is off
		st.session_state.question_order = list(range(len(all_questions)))
		st.session_state.shuffle_enabled = False

	ordered_questions = [all_questions[i] for i in st.session_state.question_order]

	# Stable one-time shuffle for options per question
	def build_option_order(options_len: int):
		order = list(range(options_len))
		random.shuffle(order)
		return order

	if shuffle_options:
		needs_reset = (not st.session_state.shuffle_options_enabled) or (len(st.session_state.option_orders) != len(ordered_questions))
		if needs_reset:
			st.session_state.option_orders = {}
			for idx, q in enumerate(ordered_questions):
				# dedupe once before computing order to keep stable length
				opts = q.get('options', [])
				opts = list(dict.fromkeys(opts))
				q['options'] = opts
				opt_len = len(opts)
				st.session_state.option_orders[f"q{idx+1}"] = build_option_order(opt_len) if opt_len > 0 else []
			st.session_state.shuffle_options_enabled = True
	else:
		st.session_state.option_orders = {}
		st.session_state.shuffle_options_enabled = False

	# Session state
	if 'quiz_submitted' not in st.session_state:
		st.session_state.quiz_submitted = False
	if 'user_answers' not in st.session_state:
		st.session_state.user_answers = {}

	# Visual separator between options and questions
	st.markdown("---")
	st.markdown(f"### 📝 Quiz ({len(all_questions)} questions)")
	if not st.session_state.quiz_submitted:
		for i, q in enumerate(ordered_questions):
			# apply option ordering if present
			opt_order = st.session_state.option_orders.get(f"q{i+1}")
			opts = q.get('options', [])
			if opt_order and opts:
				# if mismatch, rebuild a safe order on the fly
				if len(opt_order) != len(opts):
					opt_order = list(range(len(opts)))
				q = dict(q)
				q['options'] = [opts[j] for j in opt_order]
			ans = display_question(q, i + 1, len(all_questions))
			st.session_state.user_answers[f"q{i+1}"] = ans
		if st.button("Submit Quiz", type="primary", use_container_width=True):
			st.session_state.quiz_submitted = True
			st.rerun()
	else:
		score = calculate_score(ordered_questions, st.session_state.user_answers)
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Correct Answers", f"{score['correct_answers']}/{score['total_questions']}", f"{score['percentage']:.1f}%")
		with col2:
			st.metric("Points Earned", f"{score['earned_points']}/{score['total_possible_points']}", f"{score['points_percentage']:.1f}%")
		with col3:
			grade = "A" if score['percentage'] >= 90 else "B" if score['percentage'] >= 80 else "C" if score['percentage'] >= 70 else "D" if score['percentage'] >= 60 else "F"
			st.metric("Grade", grade)
		st.markdown("### 📝 Detailed Results")
		for r in score['results']:
			with st.expander(f"Question {r['question_num']}: {'✅' if r['is_correct'] else '❌'}"):
				st.write(f"**Question:** {r['question_text']}")
				st.write(f"**Your Answer:** {r['user_answer']}")
				st.write(f"**Correct Answer:** {r['correct_answer']}")
				st.write(f"**Points:** {r['points']} point{'s' if r['points'] > 1 else ''}")
		if st.button("Take Quiz Again", type="secondary", use_container_width=True):
			st.session_state.quiz_submitted = False
			st.session_state.user_answers = {}
			st.rerun()

	# Footer credit line
	st.markdown("---")
	st.markdown("<div style='text-align:center;color:#6b7280;font-size:0.9rem;'>All credit for course content: <a href='https://onlinecourses.nptel.ac.in/noc25_cs161/preview' target='_blank'>NPTEL - Introduction to Large Language Models (LLMs)</a></div>", unsafe_allow_html=True)


if __name__ == '__main__':
	main()


