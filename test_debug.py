#!/usr/bin/env python3
"""
Test the tool output directly to debug the structure issue
"""

import json
from refiner_agent.tools import retrieve_final_output_from_state

class MockContext:
    """Mock context for testing"""
    def __init__(self, state):
        self.state = state

# Create test data matching what's in state
test_state = {
    "iterations": [
        {
            "iteration": 1,
            "answer": {
                "situation": "As a software engineer at a startup, I was developing a new microservices architecture when our system started experiencing intermittent crashes during peak usage hours.",
                "task": "I needed to identify the root cause of the crashes, which were affecting customer experience and potentially costing the company thousands of dollars per hour in lost revenue.",
                "action": "I immediately implemented comprehensive monitoring across all services, including APM tools, log aggregation, and distributed tracing. I analyzed patterns in the crash logs and noticed memory spikes coinciding with specific API calls. Using profiling tools, I discovered a memory leak in our caching layer where objects weren't being properly garbage collected. I wrote a patch to fix the memory management issue, implemented proper cache eviction policies, and added automated tests to prevent similar issues.",
                "result": "The crashes stopped completely after deploying the fix. System stability improved by 99.9%, and I received recognition from the CTO for preventing potential revenue loss. This experience also led to establishing better monitoring practices across the engineering team."
            },
            "critique": {
                "rating": 4.2,
                "suggestions": ["Consider mentioning specific tools", "Add quantified impact"],
                "structure_feedback": "Well-structured STAR response.",
                "relevance_feedback": "Highly relevant for software engineering role.",
                "specificity_feedback": "Good technical detail.",
                "professional_impact_feedback": "Shows technical expertise."
            }
        },
        {
            "iteration": 2,
            "answer": {
                "situation": "In 2022, as a senior software engineer at TechStartup (a fintech company), our payment processing system began experiencing cascading failures during Black Friday, affecting over 10,000 concurrent users.",
                "task": "I needed to diagnose and fix the critical system failure within minutes to prevent significant revenue loss (estimated at $50,000 per hour) and maintain customer trust during our peak sales period.",
                "action": "I immediately accessed our Datadog monitoring dashboard and identified memory usage spiking to 95% on our Redis cache clusters. Using New Relic APM, I traced the issue to inefficient queries in our payment validation service. I quickly implemented a hotfix using connection pooling to reduce Redis connections from 5,000 to 500, deployed it through our CI/CD pipeline (Jenkins), and added circuit breakers to prevent cascading failures. I also coordinated with the DevOps team to auto-scale our infrastructure.",
                "result": "System recovery completed within 18 minutes, reducing potential losses by 70%. Payment success rate returned to 99.8% from the 23% crash rate. The incident led to implementing improved monitoring alerts and establishing an on-call rotation system. I received the company's 'Crisis Management Award' and a 15% performance bonus."
            },
            "critique": {
                "rating": 4.9,
                "suggestions": ["Nearly perfect response"],
                "structure_feedback": "Excellent STAR structure with compelling narrative.",
                "relevance_feedback": "Perfectly tailored for software engineering position.",
                "specificity_feedback": "Outstanding technical specificity with tools and metrics.",
                "professional_impact_feedback": "Demonstrates exceptional crisis management."
            }
        }
    ],
    "current_iteration": 2,
    "highest_rating": 4.9,
    "highest_rated_iteration": 2,
    "final_status": "COMPLETED_HIGH_RATING"
}

# Create mock context
mock_ctx = MockContext(test_state)

# Call the function
result = retrieve_final_output_from_state(mock_ctx)

# Print the result
print("Result structure:")
print(json.dumps(result, indent=2, default=str))

# Check specifically what's in retrieved_output
if "retrieved_output" in result:
    output = result["retrieved_output"]
    print("\nretrieved_output keys:", list(output.keys()))
    print("\nHistory length:", len(output.get("history", [])))
    
    if output.get("history"):
        print("\nFirst history item:")
        print(json.dumps(output["history"][0], indent=2, default=str))