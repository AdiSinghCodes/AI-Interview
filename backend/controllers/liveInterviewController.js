const { generateLocalQuestionBank, enrichQuestionBank } = require('../services/bankGenerator');
const Interview = require('../models/Interview');
const User = require('../models/User');
const llm = require('../services/interviewLLMService');

// ============================================================
// HELPERS
// ============================================================

function asArray(value) {
  if (Array.isArray(value)) {
    return value
      .map(String)
      .map(x => x.trim())
      .filter(Boolean);
  }

  if (
    value === undefined ||
    value === null ||
    value === ''
  ) {
    return [];
  }

  return String(value)
    .split(/[,;+]/)
    .map(x => x.trim())
    .filter(Boolean);
}

// ============================================================
// WAIT / RETRY FOR PYTHON AI AGENT
// ============================================================
//
// Python interview.py can take several seconds to start.
// Node may receive the frontend request before port 5100
// is ready.
//
// This helper retries only connection-refused errors.
// Other errors are immediately passed through.
//

async function callWithRetry(
  fn,
  options = {}
) {
  const maxAttempts =
    Number(options.maxAttempts || 30);

  const delayMs =
    Number(options.delayMs || 1000);

  for (
    let attempt = 1;
    attempt <= maxAttempts;
    attempt++
  ) {
    try {
      return await fn();
    } catch (error) {
      const code =
        error?.code ||
        error?.cause?.code;

      const isConnectionRefused =
        code === 'ECONNREFUSED' ||
        String(error?.message || '')
          .includes('ECONNREFUSED');

      // If this is not a startup/network
      // connection problem, don't hide it.
      if (!isConnectionRefused) {
        throw error;
      }

      if (
        attempt >= maxAttempts
      ) {
        throw error;
      }

      console.log(
        `AI Interview Agent not ready. ` +
        `Retrying ${attempt}/${maxAttempts} ` +
        `in ${delayMs}ms...`
      );

      await new Promise(
        resolve =>
          setTimeout(
            resolve,
            delayMs
          )
      );
    }
  }

  throw new Error(
    'AI Interview Agent could not be reached.'
  );
}

// ============================================================
// SETUP NORMALIZATION
// ============================================================

function cleanSetup(
  body = {},
  user = null
) {
  const types = asArray(
    body.types ??
    body.interview_types ??
    body.interviewTypes
  );

  const normalizedTypes =
    types.length
      ? types
      : asArray(
          body.interviewType
        );

  const profile =
    body.profile ||
    user?.profile ||
    null;

  const useResume = Boolean(
    body.useResume ??
    body.use_resume
  );

  const setup = {
    // --------------------------------------------------------
    // CORE INTERVIEW SETUP
    // --------------------------------------------------------

    domain: String(
      body.domain ??
      body.primary_domain ??
      ''
    ).trim(),

    subDomain: String(
      body.subDomain ??
      body.sub_domain ??
      body.sub ??
      ''
    ).trim(),

    role: String(
      body.role ??
      body.specific_role ??
      ''
    ).trim(),

    objective: String(
      body.objective ??
      body.interview_objective ??
      ''
    ).trim(),

    // --------------------------------------------------------
    // INTERVIEW TYPES
    // --------------------------------------------------------

    types: normalizedTypes,

    interview_types:
      normalizedTypes,

    interviewType: String(
      body.interviewType ??
      normalizedTypes.join(', ') ??
      ''
    ).trim(),

    // --------------------------------------------------------
    // INTERVIEW DETAILS
    // --------------------------------------------------------

    stage: String(
      body.stage ??
      ''
    ).trim(),

    companyType: String(
      body.companyType ??
      body.company_type ??
      ''
    ).trim(),

    company: String(
      body.company ??
      body.company_name ??
      ''
    ).trim(),

    duration: Number(
      body.duration ??
      body.duration_min ??
      45
    ),

    difficulty: String(
      body.difficulty ??
      ''
    ).trim(),

    language: String(
      body.language ??
      'English'
    ).trim(),

    // --------------------------------------------------------
    // RESUME
    // --------------------------------------------------------

    useResume,

    use_resume: useResume,

    // --------------------------------------------------------
    // CUSTOM TOPICS
    // --------------------------------------------------------

    customTopics:
      Array.isArray(
        body.customTopics ??
        body.custom_topics
      )
        ? (
            body.customTopics ??
            body.custom_topics
          )
        : String(
            body.customTopics ??
            body.custom_topics ??
            ''
          ).trim(),

    // --------------------------------------------------------
    // CODING / SQL
    // --------------------------------------------------------

    codingLanguage: String(
      body.codingLanguage ??
      body.coding_language ??
      'python'
    ).trim(),

    requiresCoding: Boolean(
      body.requiresCoding ??
      body.requires_coding
    ),

    requires_coding: Boolean(
      body.requiresCoding ??
      body.requires_coding
    ),

    requiresSql: Boolean(
      body.requiresSql ??
      body.requires_sql
    ),

    requires_sql: Boolean(
      body.requiresSql ??
      body.requires_sql
    ),

    isTechnical: Boolean(
      body.isTechnical ??
      body.is_technical
    ),

    is_technical: Boolean(
      body.isTechnical ??
      body.is_technical
    ),

    techStack:
      body.techStack ??
      body.tech_stack ??
      [],

    // --------------------------------------------------------
    // PROFILE
    // --------------------------------------------------------

    profile,

    // --------------------------------------------------------
    // IMPORTANT:
    // Resume context is ONLY included when the user selected
    // "Use Resume".
    // --------------------------------------------------------

    resume: useResume
      ? (
          body.resume ||
          profile?.resume ||
          null
        )
      : null,
  };

  return setup;
}

// ============================================================
// CLASSIFY INTERVIEW
// ============================================================

function classifyInterview(
  setup
) {
  const types =
    setup.types.map(
      x => x.toLowerCase()
    );

  const technical =
    setup.isTechnical ||
    types.some(
      t =>
        /technical|coding\s*\/\s*dsa|sql\s*\/\s*database|system design/i.test(
          t
        )
    );

  const wantsCoding =
    setup.requiresCoding ||
    types.some(
      t =>
        /technical|coding|dsa/i.test(
          t
        )
    );

  const wantsSql =
    setup.requiresSql ||
    types.some(
      t =>
        /technical|sql|database/i.test(
          t
        )
    );

  return {
    technical,
    wantsCoding,
    wantsSql,
  };
}

// ============================================================
// BUILD INTERVIEW PLAN
// ============================================================

function buildPlan(
  setup
) {
  const total =
    Math.max(
      6,
      Math.round(
        Number(
          setup.duration || 45
        ) / 3
      )
    );

  const {
    technical,
    wantsCoding,
    wantsSql,
  } =
    classifyInterview(
      setup
    );

  // ----------------------------------------------------------
  // NORMAL / NON-TECHNICAL INTERVIEW
  // ----------------------------------------------------------

  if (!technical) {
    return {
      total,
      verbal: total,
      coding: 0,
      sql: 0,
    };
  }

  // ----------------------------------------------------------
  // TECHNICAL INTERVIEW
  // Roughly 60% verbal + 40% technical
  // ----------------------------------------------------------

  const technicalUnits =
    Math.max(
      2,
      Math.round(
        total * 0.4
      )
    );

  const verbal =
    total -
    technicalUnits;

  let sql =
    wantsSql
      ? Math.max(
          1,
          Math.round(
            technicalUnits *
              0.4
          )
        )
      : 0;

  let coding =
    wantsCoding
      ? technicalUnits - sql
      : 0;

  if (
    !wantsCoding &&
    wantsSql
  ) {
    sql =
      technicalUnits;

    coding = 0;
  }

  if (
    wantsCoding &&
    !wantsSql
  ) {
    coding =
      technicalUnits;

    sql = 0;
  }

  return {
    total,
    verbal,
    coding,
    sql,
  };
}

// ============================================================
// CONVERT MONGODB HISTORY TO LLM HISTORY
// ============================================================

function historyFrom(
  interview
) {
  return interview.answers.map(
    a => ({
      section: a.section,
      question: a.question,
      questionPayload:
        a.questionPayload,
      answer: a.answer,
      transcript: a.transcript,
      evaluation:
        a.evaluation,
      score: a.score,
    })
  );
}

// ============================================================
// AVERAGE SCORE
// ============================================================

function averageScore(
  answers
) {
  const scored =
    answers
      .map(
        a => Number(a.score)
      )
      .filter(
        Number.isFinite
      );

  return scored.length
    ? Number(
        (
          scored.reduce(
            (a, b) =>
              a + b,
            0
          ) /
          scored.length
        ).toFixed(2)
      )
    : 0;
}

// ============================================================
// START INTERVIEW
// ============================================================

exports.start =
  async (req, res) => {
    try {
      // ------------------------------------------------------
      // LOAD USER
      // ------------------------------------------------------

      const user =
        await User.findById(
          req.userId
        ).lean();

      if (!user) {
        return res
          .status(404)
          .json({
            message:
              'User not found.',
          });
      }

      if (
        !user.profileCompleted
      ) {
        return res
          .status(403)
          .json({
            message:
              'Complete your profile first.',
          });
      }

      // ------------------------------------------------------
      // NORMALIZE SETUP
      // ------------------------------------------------------

      const setup =
        cleanSetup(
          req.body,
          user
        );

      if (
        !setup.domain ||
        !setup.subDomain ||
        !setup.role ||
        !setup.objective ||
        !setup.types.length
      ) {
        return res
          .status(400)
          .json({
            message:
              'Domain, subdomain, role, objective and interview type are required.',
          });
      }

      // ------------------------------------------------------
      // BUILD PLAN
      // ------------------------------------------------------

      const plan =
        buildPlan(setup);

      // ------------------------------------------------------
      // CREATE INTERVIEW
      // ------------------------------------------------------

      const interview =
        await Interview.create({
          userId:
            req.userId,

          setup,

          status:
            'active',

          startedAt:
            new Date(),

          questionCount:
            plan.total,
        });

      // ------------------------------------------------------
      // ASK PYTHON AI AGENT
      //
      // IMPORTANT:
      // Retry if Python 5100 hasn't started yet.
      // ------------------------------------------------------

      // ------------------------------------------------------
      // PRE-GENERATE 3-TIER QUESTION BANK (EASY -> MEDIUM -> HARD)
      // ------------------------------------------------------
      let questionBank = [];
      try {
        const bankRes = await callWithRetry(
          () => llm.generateQuestionBank({ setup, plan }),
          { maxAttempts: 5, delayMs: 1000 }
        );
        if (Array.isArray(bankRes?.questions) && bankRes.questions.length > 0) {
          questionBank = bankRes.questions;
        }
      } catch (bankErr) {
        console.error('Python question bank endpoint warning, using local generator fallback:', bankErr?.message || bankErr);
      }

      // ENFORCE PSEUDOCODE AND CODING/SQL WORKSPACE PROBLEMS FOR TECHNICAL INTERVIEWS
      questionBank = enrichQuestionBank(questionBank, setup, plan);

      const q = questionBank[0];

      interview.questionBank = questionBank;
      await interview.save();

      return res.json({
        interviewId: String(interview._id),
        setup,
        plan,
        question: q,
        questionBank,
      });
    } catch (e) {
      console.error(
        'Start interview failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to start interview',
        });
    }
  };

// ============================================================
// ANSWER INTERVIEW QUESTION
// ============================================================

exports.answer =
  async (req, res) => {
    try {
      // ------------------------------------------------------
      // LOAD INTERVIEW
      // ------------------------------------------------------

      const interview =
        await Interview.findOne({
          _id:
            req.params.id,

          userId:
            req.userId,
        });

      if (!interview) {
        return res
          .status(404)
          .json({
            message:
              'Interview not found.',
          });
      }

      if (
        interview.status !==
        'active'
      ) {
        return res
          .status(400)
          .json({
            message:
              'Interview is not active.',
          });
      }

      // ------------------------------------------------------
      // PLAN + HISTORY
      // ------------------------------------------------------

      const plan =
        buildPlan(
          interview.setup
        );

      const history =
        historyFrom(
          interview
        );

      // ------------------------------------------------------
      // CURRENT ANSWER
      // ------------------------------------------------------

      const question =
        String(
          req.body.question ||
          ''
        ).trim();

      const answer =
        String(
          req.body.answer ??
          req.body.transcript ??
          ''
        ).trim();

      const transcript =
        String(
          req.body.transcript ??
          answer
        ).trim();

      const section =
        [
          'verbal',
          'coding',
          'sql',
        ].includes(
          req.body.section
        )
          ? req.body.section
          : 'verbal';

      const codingSubmission =
        req.body
          .codingSubmission ||
        null;

      if (!question) {
        return res
          .status(400)
          .json({
            message:
              'Question is required.',
          });
      }

      if (
        !answer &&
        !codingSubmission
      ) {
        return res
          .status(400)
          .json({
            message:
              'Answer is required.',
          });
      }

      // ------------------------------------------------------
      // EVALUATION PAYLOAD
      // ------------------------------------------------------

      const payload = {
        setup:
          interview.setup,

        plan,

        history,

        questionIndex:
          interview.answers
            .length + 1,

        currentQuestion:
          question,

        answer,

        transcript,

        section,

        codingSubmission,
      };

      // ------------------------------------------------------
      // EVALUATE ANSWER
      //
      // Retry because the Python agent can temporarily
      // restart or take time to become available.
      // ------------------------------------------------------

      const result =
        await callWithRetry(
          () =>
            llm.evaluateAnswer(
              payload
            ),
          {
            maxAttempts: 30,
            delayMs: 1000,
          }
        );

      const evaluation =
        result?.evaluation ||
        result ||
        {};

      const score =
        Number(
          result?.score ??
          evaluation?.score ??
          0
        );

      const followUp =
        Boolean(
          result?.followUp ??
          evaluation?.followUp
        );

      const followUpReason =
        String(
          result?.followUpReason ??
          evaluation?.followUpReason ??
          ''
        );

      // ------------------------------------------------------
      // SAVE ANSWER
      // ------------------------------------------------------

      const isCrossQuestion = Boolean(
        req.body.crossQuestion ||
        req.body.questionPayload?.crossQuestion ||
        req.body.questionType === 'cross_question' ||
        req.body.questionPayload?.questionType === 'cross_question' ||
        /cross-question|follow-up/i.test(req.body.questionPayload?.category || '') ||
        /cross-question|follow-up/i.test(req.body.category || '')
      );

      const savedAnswer = {
        questionNumber: interview.answers.length + 1,
        section,
        questionType: isCrossQuestion ? 'cross_question' : (result?.questionType || section),
        crossQuestion: isCrossQuestion,
        question,
        questionPayload: req.body.questionPayload || null,
        answer,
        transcript,
        codingSubmission,
        followUp,
        followUpReason,
        evaluation,
        score,
        answeredAt: new Date(),
      };

      interview.answers.push(
        savedAnswer
      );

      // ------------------------------------------------------
      // SHOULD INTERVIEW END?
      // ------------------------------------------------------

      const reachedPlan =
        interview.answers
          .length >=
        plan.total;

      const shouldEnd =
        reachedPlan ||
        Boolean(
          result?.endInterview ??
          evaluation?.endInterview
        );

      // ------------------------------------------------------
      // COMPLETE INTERVIEW
      // ------------------------------------------------------

      if (shouldEnd) {
        interview.status =
          'completed';

        interview.completedAt =
          new Date();

        interview.durationSeconds =
          interview.startedAt
            ? Math.round(
                (
                  Date.now() -
                  interview.startedAt.getTime()
                ) / 1000
              )
            : 0;

        interview.finalScore =
          averageScore(
            interview.answers
          );

        interview.summary = {
          averageScore:
            interview.finalScore,

          questionCount:
            interview.answers.length,

          completedBecause:
            reachedPlan
              ? 'planned_question_count'
              : 'ai_end_interview',

          lastSummary:
            result?.summary ??
            evaluation?.summary ??
            '',
        };

        await interview.save();

        return res.json({
          done: true,

          evaluation,

          score,

          summary:
            interview.summary,

          interview:
            interview.toObject(),
        });
      }

      // ------------------------------------------------------
      // NEXT QUESTION
      // ------------------------------------------------------

      let next;

      // ------------------------------------------------------
      // CROSS QUESTION
      // ------------------------------------------------------

      // ------------------------------------------------------
      // NEXT QUESTION SELECTION (PREVENTS REPEATING QUESTION LOOPS)
      // ------------------------------------------------------
      // Count main (non-cross) questions answered so far
      const mainQuestionsCount = interview.answers.filter(
        a => a.questionType !== 'cross_question' && !a.crossQuestion
      ).length;

      // Check if previous question was ALREADY a cross-question
      const lastAns = interview.answers[interview.answers.length - 1];
      const lastWasCross = Boolean(
        lastAns && (
          lastAns.questionType === 'cross_question' || 
          lastAns.crossQuestion || 
          /cross-question|follow-up/i.test(lastAns.category || '')
        )
      );

      // Check if candidate explicitly opted out or said "I don't know"
      const candidateOptedOut = /i don['’]?t know|no idea|idk|pass|don['’]?t know/i.test(answer);

      // Allow at most ONE follow-up per main question, blocked if candidate opted out or last was cross
      const allowFollowUp = Boolean(
        followUp && 
        !lastWasCross && 
        !candidateOptedOut && 
        (result?.followUpQuestion || evaluation?.followUpQuestion)
      );

      if (allowFollowUp) {
        const currentTier = (Array.isArray(interview.questionBank) && interview.questionBank[mainQuestionsCount - 1]?.difficultyTier) || 'easy';
        next = {
          section: 'verbal',
          questionType: 'cross_question',
          difficultyTier: currentTier,
          text: String(result.followUpQuestion || evaluation.followUpQuestion),
          category: `${currentTier.charAt(0).toUpperCase() + currentTier.slice(1)} / Follow-up`,
          crossQuestion: true,
        };
        console.log(`🎯 Triggered single follow-up for Question ${mainQuestionsCount}`);
      } else {
        // ----------------------------------------------------
        // PROGRESS TO NEXT MAIN QUESTION FROM 3-TIER QUESTION BANK
        // ----------------------------------------------------
        if (Array.isArray(interview.questionBank) && mainQuestionsCount < interview.questionBank.length) {
          next = interview.questionBank[mainQuestionsCount];
          console.log(`PROGRESSING QUESTION BANK (0ms delay): Question Bank Index ${mainQuestionsCount + 1}/${interview.questionBank.length} (${next.difficultyTier || 'tiered'})`);
        } else {
          const localBank = generateLocalQuestionBank(interview.setup, plan);
          next = localBank[mainQuestionsCount % localBank.length] || localBank[0];
        }
      }

      // ------------------------------------------------------
      // SAVE INTERVIEW
      // ------------------------------------------------------

      await interview.save();

      // ------------------------------------------------------
      // RESPONSE
      // ------------------------------------------------------

      return res.json({
        done: false,

        evaluation,

        score,

        followUp,

        followUpReason,

        nextQuestion:
          next,
      });
    } catch (e) {
      console.error(
        'Answer processing failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to process answer',
        });
    }
  };

// ============================================================
// TRANSCRIBE ANSWER AUDIO
// ============================================================

exports.transcribe =
  async (req, res) => {
    try {
      if (!req.file) {
        return res
          .status(400)
          .json({
            message:
              'Audio file is required.',
          });
      }

      // Audio is only passed to Whisper.
      // It is not saved as an interview recording.

      const result =
        await llm.transcribe(
          req.file.buffer,
          req.file.originalname,
          req.file.mimetype
        );

      return res.json(
        result
      );
    } catch (e) {
      console.error(
        'Transcription failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Transcription failed',
        });
    }
  };

// ============================================================
// COMPLETE INTERVIEW MANUALLY
// ============================================================

exports.complete =
  async (req, res) => {
    try {
      const interview =
        await Interview.findOne({
          _id:
            req.params.id,

          userId:
            req.userId,
        });

      if (!interview) {
        return res
          .status(404)
          .json({
            message:
              'Interview not found.',
          });
      }

      if (
        interview.status ===
        'active'
      ) {
        interview.status =
          'completed';

        interview.completedAt =
          new Date();

        interview.durationSeconds =
          interview.startedAt
            ? Math.round(
                (
                  Date.now() -
                  interview.startedAt.getTime()
                ) / 1000
              )
            : 0;

        interview.finalScore =
          averageScore(
            interview.answers
          );

        interview.summary = {
          averageScore:
            interview.finalScore,

          questionCount:
            interview.answers.length,

          completedBecause:
            'candidate_ended_interview',
        };

        await interview.save();
      }

      return res.json({
        done: true,

        interview:
          interview.toObject(),
      });
    } catch (e) {
      console.error(
        'Complete interview failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to complete interview',
        });
    }
  };

// ============================================================
// LIST COMPLETED INTERVIEWS / REPORTS
// ============================================================

exports.list =
  async (req, res) => {
    try {
      const interviews =
        await Interview.find({
          userId:
            req.userId,

          status:
            'completed',
        })
          .sort({
            completedAt: -1,
            createdAt: -1,
          })
          .lean();

      return res.json({
        interviews,
      });
    } catch (e) {
      console.error(
        'Load reports failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to load reports',
        });
    }
  };

// ============================================================
// GET SINGLE INTERVIEW
// ============================================================

exports.get =
  async (req, res) => {
    try {
      const interview =
        await Interview.findOne({
          _id:
            req.params.id,

          userId:
            req.userId,
        }).lean();

      if (!interview) {
        return res
          .status(404)
          .json({
            message:
              'Interview not found.',
          });
      }

      return res.json(
        interview
      );
    } catch (e) {
      console.error(
        'Get interview failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to get interview',
        });
    }
  };

// ============================================================
// EXPORT HELPERS
// ============================================================

module.exports.cleanSetup =
  cleanSetup;

module.exports.buildPlan =
  buildPlan;

module.exports.callWithRetry =
  callWithRetry;