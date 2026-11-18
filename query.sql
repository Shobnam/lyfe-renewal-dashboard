WITH latest_patient_ext AS (
    SELECT DISTINCT ON (pei.patient_id)
        pei.patient_id,
        pei.patient_type,
        pei.patient_tag,
        pei.engagement_status,
        pei.updated_on
    FROM patient_ext_info pei
    ORDER BY
        pei.patient_id,
        pei.updated_on DESC NULLS LAST
),

active_hc AS (
    SELECT DISTINCT ON (hc.patient_id)
        hc.patient_id,
        NULLIF(TRIM(hc.coach_name), '') AS hc_name
    FROM assignedcoach hc
    WHERE UPPER(
        TRIM(COALESCE(hc.assignment_status, ''))
    ) = 'ACTIVE'
    ORDER BY
        hc.patient_id,
        hc.id DESC
),

active_lcm AS (
    SELECT DISTINCT ON (lcm.patient_id)
        lcm.patient_id,

        CONCAT_WS(
            ' ',
            NULLIF(TRIM(lcm.ace_first_name), ''),
            NULLIF(TRIM(lcm.ace_last_name), '')
        ) AS lcm_name

    FROM assigned_ace lcm

    WHERE UPPER(
        TRIM(COALESCE(lcm.status, ''))
    ) = 'ACTIVE'

    ORDER BY
        lcm.patient_id,
        lcm.updated_on DESC NULLS LAST,
        lcm.id DESC
),

renewal_patients AS (
    SELECT
        p.patient_id,

        p.identifier::TEXT AS identifier,

        CONCAT_WS(
            ' ',
            NULLIF(TRIM(p.firstname), ''),
            NULLIF(TRIM(p.lastname), '')
        ) AS patient_name,

        p.email,

        TO_TIMESTAMP(
            p.subscription_start_dt / 1000.0
        )::date AS subscription_start_date,

        TO_TIMESTAMP(
            p.subscription_end_dt / 1000.0
        )::date AS subscription_end_date,

        lpe.engagement_status,

        hc.hc_name,

        lcm.lcm_name

    FROM patient p

    LEFT JOIN latest_patient_ext lpe
        ON lpe.patient_id = p.patient_id

    LEFT JOIN active_hc hc
        ON hc.patient_id = p.patient_id

    LEFT JOIN active_lcm lcm
        ON lcm.patient_id = p.patient_id

    WHERE
        TO_TIMESTAMP(
            p.subscription_end_dt / 1000.0
        )::date >= :month_start

        AND TO_TIMESTAMP(
            p.subscription_end_dt / 1000.0
        )::date < :next_month_start

        AND COALESCE(p.is_deleted, FALSE) = FALSE

        AND p.deleted_on IS NULL

        AND NULLIF(
            TRIM(COALESCE(p.identifier::TEXT, '')),
            ''
        ) IS NOT NULL

        AND UPPER(
            TRIM(COALESCE(p.patient_type, ''))
        ) <> 'PDC'

        AND UPPER(
            TRIM(COALESCE(p.patient_tag, ''))
        ) <> 'PDC'

        AND UPPER(
            TRIM(COALESCE(lpe.patient_type, ''))
        ) <> 'PDC'

        AND UPPER(
            TRIM(COALESCE(lpe.patient_tag, ''))
        ) <> 'PDC'

        AND UPPER(
            TRIM(COALESCE(p.patient_tag, ''))
        ) <> 'TEST'

        AND UPPER(
            TRIM(COALESCE(lpe.patient_tag, ''))
        ) <> 'TEST'

        AND LOWER(
            COALESCE(p.identifier::TEXT, '')
        ) NOT LIKE '%test%'

        AND LOWER(
            COALESCE(p.firstname, '')
        ) NOT LIKE '%test%'

        AND LOWER(
            COALESCE(p.lastname, '')
        ) NOT LIKE '%test%'

        AND LOWER(
            COALESCE(p.email, '')
        ) NOT LIKE '%test%'

        AND UPPER(
            TRIM(COALESCE(lpe.engagement_status, ''))
        ) NOT LIKE '%ABORTED%'

        AND UPPER(
            TRIM(COALESCE(lpe.engagement_status, ''))
        ) NOT LIKE '%STOPPED%'

        AND UPPER(
            TRIM(COALESCE(lpe.engagement_status, ''))
        ) NOT LIKE '%DECEASED%'

        AND LOWER(
            COALESCE(hc.hc_name, '')
        ) NOT LIKE '%test%'

        AND LOWER(
            COALESCE(lcm.lcm_name, '')
        ) NOT LIKE '%test%'
)

SELECT
    identifier AS "Identifier",

    patient_name AS "Patient Name",

    email AS "Email",

    subscription_start_date
        AS "Subscription Start Date",

    subscription_end_date
        AS "Subscription End Date",

    subscription_end_date - CURRENT_DATE
        AS "Days Remaining",

    engagement_status
        AS "Latest Engagement Status",

    hc_name
        AS "HC Name",

    lcm_name
        AS "LCM Name"

FROM renewal_patients

ORDER BY
    subscription_end_date ASC,
    identifier ASC;
