-- Enhanced materialized view with complete management chain for each person
CREATE MATERIALIZED VIEW enhanced_org_hierarchy_mv AS
WITH RECURSIVE 
-- Unified view of all personnel (employees and contractors)
all_personnel AS (
    -- Regular employees
    SELECT 
        empid AS person_id,
        name AS person_name,
        mgrid AS manager_id,
        'Employee' AS person_type
    FROM 
        emp
    
    UNION ALL
    
    -- Contractors
    SELECT 
        contractor_id AS person_id,
        contractor_name AS person_name,
        emp_supervisor_id AS manager_id,
        'Contractor' AS person_type
    FROM 
        contractor
),

-- Build the complete hierarchy with management chain
org_chain AS (
    -- Base case: Start with people who don't have managers (top executives)
    -- Or alternatively, start with people at a specific level
    SELECT 
        p.person_id,
        p.person_name,
        p.manager_id,
        p.person_type,
        1 AS level,
        ARRAY[p.person_id] AS chain,
        ARRAY[p.person_name] AS name_chain
    FROM 
        all_personnel p
    WHERE 
        p.manager_id IS NULL  -- Top executives
        OR p.manager_id NOT IN (SELECT person_id FROM all_personnel)  -- People whose managers aren't in our data
    
    UNION ALL
    
    -- Recursive case: Find all personnel reporting to anyone in the chain
    SELECT 
        p.person_id,
        p.person_name,
        p.manager_id,
        p.person_type,
        oc.level + 1 AS level,
        oc.chain || p.person_id AS chain,  -- Append this person to the management chain
        oc.name_chain || p.person_name AS name_chain  -- Append this person's name to the chain
    FROM 
        all_personnel p
    JOIN 
        org_chain oc ON p.manager_id = oc.person_id
)

-- Final materialized view with all management levels identified
SELECT 
    oc.person_id,
    oc.person_name,
    oc.manager_id,
    oc.person_type,
    oc.level,
    m.person_name AS manager_name,
    
    -- SVP details (assuming level 1 is SVP)
    CASE WHEN oc.level >= 3 THEN oc.chain[1] ELSE NULL END AS svp_id,
    CASE WHEN oc.level >= 3 THEN oc.name_chain[1] ELSE NULL END AS svp_name,
    
    -- VP details (assuming level 2 is VP)
    CASE WHEN oc.level >= 4 THEN oc.chain[2] ELSE NULL END AS vp_id,
    CASE WHEN oc.level >= 4 THEN oc.name_chain[2] ELSE NULL END AS vp_name,
    
    -- Sr. Director details (assuming level 3 is Sr. Director)
    CASE WHEN oc.level >= 5 THEN oc.chain[3] ELSE NULL END AS sr_director_id,
    CASE WHEN oc.level >= 5 THEN oc.name_chain[3] ELSE NULL END AS sr_director_name,
    
    -- Director details (assuming level 4 is Director)
    CASE WHEN oc.level >= 6 THEN oc.chain[4] ELSE NULL END AS director_id,
    CASE WHEN oc.level >= 6 THEN oc.name_chain[4] ELSE NULL END AS director_name,
    
    -- Store the entire chain for potential future use
    oc.chain AS management_chain,
    oc.name_chain AS management_name_chain
FROM 
    org_chain oc
LEFT JOIN 
    all_personnel m ON oc.manager_id = m.person_id
WITH DATA;

-- Create indexes to improve query performance
CREATE INDEX idx_enhanced_org_person_id ON enhanced_org_hierarchy_mv(person_id);
CREATE INDEX idx_enhanced_org_person_name ON enhanced_org_hierarchy_mv(person_name);
CREATE INDEX idx_enhanced_org_svp_id ON enhanced_org_hierarchy_mv(svp_id);
CREATE INDEX idx_enhanced_org_vp_id ON enhanced_org_hierarchy_mv(vp_id);
CREATE INDEX idx_enhanced_org_sr_dir_id ON enhanced_org_hierarchy_mv(sr_director_id);
