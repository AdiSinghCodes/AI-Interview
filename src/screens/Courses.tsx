import React, { useEffect, useMemo, useState } from "react";
import {
  Search,
  BookOpen,
  ExternalLink,
  Loader2,
  RefreshCw,
  PlayCircle,
  GraduationCap,
  Clock,
  Sparkles,
  Globe,
} from "lucide-react";

import { api } from "../api/api";
import { useAuth } from "../context/AuthContext";

type Resource = {
  title: string;
  description?: string;
  url: string;
  source?: string;
  platform?: string;
  type?: "youtube" | "course" | "article" | "documentation";
  duration?: string;
  level?: string;
  skills?: string[];
  relevance?: number;
  free?: boolean;
};

const FALLBACK_RESOURCES: Resource[] = [
  {
    title: "Learn JavaScript - Full Course for Beginners",
    description:
      "A complete beginner-friendly JavaScript course from freeCodeCamp.",
    url: "https://www.youtube.com/results?search_query=freeCodeCamp+Learn+JavaScript+Full+Course",
    source: "YouTube",
    platform: "YouTube",
    type: "youtube",
    level: "Beginner",
    free: true,
  },
  {
    title: "JavaScript Courses",
    description:
      "Structured JavaScript courses and specializations from universities and industry providers.",
    url: "https://www.coursera.org/courses?query=javascript",
    source: "Coursera",
    platform: "Coursera",
    type: "course",
    level: "Beginner → Advanced",
  },
  {
    title: "Node.js Courses",
    description:
      "Courses covering Node.js, REST APIs, Express and backend development.",
    url: "https://www.coursera.org/courses?query=node.js",
    source: "Coursera",
    platform: "Coursera",
    type: "course",
    level: "Beginner → Advanced",
  },
  {
    title: "Full Stack JavaScript Learning",
    description:
      "Free resources covering JavaScript, React, Node.js, TypeScript and full-stack development.",
    url: "https://www.classcentral.com/report/javascript-online-courses/",
    source: "Class Central",
    platform: "Class Central",
    type: "course",
    level: "Beginner → Advanced",
    free: true,
  },
];

export default function Courses() {
  const { user } = useAuth();

  const [resources, setResources] =
    useState<Resource[]>([]);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [search, setSearch] =
    useState("");

  const [activeTab, setActiveTab] =
    useState<"recommended" | "youtube" | "courses">(
      "recommended"
    );

  const profile = user?.profile;

  const fullName =
    profile?.fullName ||
    `${user?.firstName || ""} ${
      user?.lastName || ""
    }`.trim() ||
    "Learner";

  const targetRole =
    profile?.targetRole ||
    profile?.desiredRole ||
    "Software Engineer";

  const skills = useMemo(() => {
    const value =
      profile?.skills ||
      profile?.technicalSkills ||
      [];

    if (Array.isArray(value)) {
      return value.filter(Boolean);
    }

    if (typeof value === "string") {
      return value
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);
    }

    return [];
  }, [profile]);

  const profileQuery = useMemo(() => {
    const importantSkills = skills
      .slice(0, 12)
      .join(", ");

    return `${targetRole} ${importantSkills} learning courses tutorials`;
  }, [targetRole, skills]);

  const fetchResources = async (
    customQuery?: string
  ) => {
    setLoading(true);
    setError("");

    try {
      const query =
        customQuery?.trim() ||
        search.trim() ||
        profileQuery;

      const result =
        await api.searchInternetCourses(query);

      /*
       Expected backend response:

       {
         resources: [
           {
             title,
             description,
             url,
             source,
             platform,
             type,
             duration,
             level,
             skills,
             relevance,
             free
           }
         ]
       }

       We also support a direct array response.
      */

      let data: any = result;

      if (result?.resources) {
        data = result.resources;
      } else if (result?.results) {
        data = result.results;
      }

      if (!Array.isArray(data)) {
        data = [];
      }

      const validResources =
        data.filter(
          (item: any) =>
            item &&
            typeof item.url === "string" &&
            /^https?:\/\//i.test(item.url)
        );

      if (validResources.length > 0) {
        setResources(validResources);
      } else {
        setResources(FALLBACK_RESOURCES);
      }
    } catch (err: any) {
      console.error(
        "Course search failed:",
        err
      );

      setError(
        "Unable to retrieve live learning resources. Showing verified fallback resources."
      );

      setResources(FALLBACK_RESOURCES);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResources();
  }, []);

  const filteredResources = useMemo(() => {
    let list = [...resources];

    if (activeTab === "youtube") {
      list = list.filter(
        (item) =>
          item.type === "youtube" ||
          item.platform
            ?.toLowerCase()
            .includes("youtube") ||
          item.source
            ?.toLowerCase()
            .includes("youtube")
      );
    }

    if (activeTab === "courses") {
      list = list.filter(
        (item) =>
          item.type === "course" ||
          item.type === "documentation" ||
          item.type === "article"
      );
    }

    if (search.trim()) {
      const q =
        search.toLowerCase();

      list = list.filter((item) => {
        const text = [
          item.title,
          item.description,
          item.source,
          item.platform,
          item.level,
          ...(item.skills || []),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        return text.includes(q);
      });
    }

    return list;
  }, [
    resources,
    activeTab,
    search,
  ]);

  const youtubeResources =
    resources.filter(
      (item) =>
        item.type === "youtube" ||
        item.platform
          ?.toLowerCase()
          .includes("youtube") ||
        item.source
          ?.toLowerCase()
          .includes("youtube")
    );

  const courseResources =
    resources.filter(
      (item) =>
        item.type !== "youtube"
    );

  return (
    <div
      style={{
        padding: "32px",
        maxWidth: "1400px",
        margin: "0 auto",
      }}
    >
      {/* HEADER */}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "20px",
          marginBottom: "28px",
        }}
      >
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              marginBottom: "8px",
            }}
          >
            <Sparkles
              size={22}
              color="#3358E8"
            />

            <span
              style={{
                color: "#3358E8",
                fontWeight: 700,
                fontSize: "13px",
              }}
            >
              AI PERSONALIZED LEARNING
            </span>
          </div>

          <h1
            style={{
              margin: 0,
              fontSize: "32px",
              fontWeight: 800,
            }}
          >
            Courses & Learning
          </h1>

          <p
            style={{
              marginTop: "8px",
              color: "#64748B",
            }}
          >
            Real courses and YouTube resources
            selected for {fullName}.
          </p>
        </div>

        <button
          onClick={() =>
            fetchResources()
          }
          disabled={loading}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            border: "1px solid #E2E8F0",
            background: "#fff",
            borderRadius: "10px",
            padding: "10px 16px",
            cursor: "pointer",
          }}
        >
          <RefreshCw
            size={16}
            className={
              loading
                ? "spin"
                : ""
            }
          />

          Refresh
        </button>
      </div>

      {/* PROFILE CONTEXT */}

      <div
        style={{
          background:
            "linear-gradient(135deg,#EEF4FF,#F8FAFF)",
          border: "1px solid #DCE6FF",
          borderRadius: "16px",
          padding: "20px",
          marginBottom: "24px",
        }}
      >
        <div
          style={{
            display: "flex",
            gap: "14px",
            alignItems: "flex-start",
          }}
        >
          <div
            style={{
              width: "42px",
              height: "42px",
              borderRadius: "12px",
              background: "#3358E8",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
            }}
          >
            <GraduationCap
              size={22}
            />
          </div>

          <div>
            <strong>
              Learning path for {targetRole}
            </strong>

            <p
              style={{
                margin:
                  "5px 0 10px",
                color: "#64748B",
                fontSize: "14px",
              }}
            >
              Resources are searched and
              ranked using your target role
              and skills.
            </p>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "7px",
              }}
            >
              {skills
                .slice(0, 12)
                .map((skill) => (
                  <span
                    key={skill}
                    style={{
                      background:
                        "#fff",
                      border:
                        "1px solid #D8E2FF",
                      borderRadius: "999px",
                      padding:
                        "5px 10px",
                      fontSize:
                        "12px",
                      color:
                        "#3358E8",
                      fontWeight: 600,
                    }}
                  >
                    {skill}
                  </span>
                ))}
            </div>
          </div>
        </div>
      </div>

      {/* SEARCH */}

      <div
        style={{
          display: "flex",
          gap: "10px",
          marginBottom: "22px",
        }}
      >
        <div
          style={{
            flex: 1,
            position: "relative",
          }}
        >
          <Search
            size={18}
            style={{
              position: "absolute",
              left: "15px",
              top: "50%",
              transform:
                "translateY(-50%)",
              color: "#94A3B8",
            }}
          />

          <input
            value={search}
            onChange={(e) =>
              setSearch(
                e.target.value
              )
            }
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                fetchResources(
                  search
                );
              }
            }}
            placeholder="Search JavaScript, React, Python, SQL, AI..."
            style={{
              width: "100%",
              boxSizing:
                "border-box",
              padding:
                "13px 15px 13px 45px",
              border:
                "1px solid #E2E8F0",
              borderRadius: "11px",
              outline: "none",
              fontSize: "14px",
            }}
          />
        </div>

        <button
          onClick={() =>
            fetchResources(
              search
            )
          }
          disabled={
            loading ||
            !search.trim()
          }
          style={{
            padding:
              "0 22px",
            border: 0,
            borderRadius: "11px",
            background:
              "#3358E8",
            color: "#fff",
            fontWeight: 700,
            cursor: "pointer",
          }}
        >
          Search
        </button>
      </div>

      {/* TABS */}

      <div
        style={{
          display: "flex",
          gap: "8px",
          borderBottom:
            "1px solid #E2E8F0",
          marginBottom: "24px",
        }}
      >
        {[
          {
            id: "recommended",
            label: "Recommended",
            icon: Sparkles,
          },
          {
            id: "youtube",
            label: "YouTube",
            icon: PlayCircle,
          },
          {
            id: "courses",
            label: "Courses",
            icon: BookOpen,
          },
        ].map((tab) => {
          const Icon =
            tab.icon;

          const active =
            activeTab ===
            tab.id;

          return (
            <button
              key={tab.id}
              onClick={() =>
                setActiveTab(
                  tab.id as any
                )
              }
              style={{
                display: "flex",
                alignItems:
                  "center",
                gap: "7px",
                padding:
                  "11px 16px",
                border: 0,
                borderBottom:
                  active
                    ? "2px solid #3358E8"
                    : "2px solid transparent",
                background:
                  "transparent",
                color: active
                  ? "#3358E8"
                  : "#64748B",
                fontWeight:
                  active
                    ? 700
                    : 500,
                cursor: "pointer",
              }}
            >
              <Icon size={16} />

              {tab.label}
            </button>
          );
        })}
      </div>

      {/* STATUS */}

      {loading && (
        <div
          style={{
            display: "flex",
            justifyContent:
              "center",
            alignItems: "center",
            gap: "10px",
            padding: "50px",
            color: "#64748B",
          }}
        >
          <Loader2
            size={22}
            className="spin"
          />

          Searching the web for
          relevant learning resources...
        </div>
      )}

      {error && (
        <div
          style={{
            padding: "12px 16px",
            background: "#FFF7ED",
            border:
              "1px solid #FED7AA",
            color: "#9A3412",
            borderRadius: "10px",
            marginBottom: "20px",
            fontSize: "13px",
          }}
        >
          {error}
        </div>
      )}

      {/* RESULTS */}

      {!loading &&
        filteredResources.length ===
          0 && (
          <div
            style={{
              textAlign: "center",
              padding: "60px",
              color: "#64748B",
            }}
          >
            No resources found.
          </div>
        )}

      {!loading && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fill,minmax(320px,1fr))",
            gap: "18px",
          }}
        >
          {filteredResources.map(
            (
              resource,
              index
            ) => {
              const isYouTube =
                resource.type ===
                  "youtube" ||
                resource.platform
                  ?.toLowerCase()
                  .includes(
                    "youtube"
                  ) ||
                resource.source
                  ?.toLowerCase()
                  .includes(
                    "youtube"
                  );

              return (
                <div
                  key={`${resource.url}-${index}`}
                  style={{
                    border:
                      "1px solid #E2E8F0",
                    borderRadius:
                      "15px",
                    background:
                      "#fff",
                    padding:
                      "20px",
                    display: "flex",
                    flexDirection:
                      "column",
                    minHeight:
                      "235px",
                  }}
                >
                  {/* SOURCE */}

                  <div
                    style={{
                      display:
                        "flex",
                      justifyContent:
                        "space-between",
                      alignItems:
                        "center",
                      marginBottom:
                        "14px",
                    }}
                  >
                    <div
                      style={{
                        display:
                          "flex",
                        alignItems:
                          "center",
                        gap: "8px",
                        fontSize:
                          "12px",
                        fontWeight:
                          700,
                        color:
                          isYouTube
                            ? "#DC2626"
                            : "#3358E8",
                      }}
                    >
                      {isYouTube ? (
                        <PlayCircle
                          size={17}
                        />
                      ) : (
                        <BookOpen
                          size={17}
                        />
                      )}

                      {resource.platform ||
                        resource.source ||
                        "Learning Resource"}
                    </div>

                    {resource.free && (
                      <span
                        style={{
                          background:
                            "#ECFDF5",
                          color:
                            "#047857",
                          borderRadius:
                            "999px",
                          padding:
                            "4px 8px",
                          fontSize:
                            "11px",
                          fontWeight:
                            700,
                        }}
                      >
                        FREE
                      </span>
                    )}
                  </div>

                  {/* TITLE */}

                  <h3
                    style={{
                      margin:
                        "0 0 9px",
                      fontSize:
                        "17px",
                      lineHeight:
                        1.4,
                    }}
                  >
                    {resource.title}
                  </h3>

                  {/* DESCRIPTION */}

                  <p
                    style={{
                      color:
                        "#64748B",
                      fontSize:
                        "13px",
                      lineHeight:
                        1.55,
                      margin:
                        "0 0 15px",
                    }}
                  >
                    {resource.description ||
                      "Relevant learning resource selected for your profile."}
                  </p>

                  {/* META */}

                  <div
                    style={{
                      display:
                        "flex",
                      gap: "12px",
                      flexWrap:
                        "wrap",
                      marginBottom:
                        "18px",
                    }}
                  >
                    {resource.level && (
                      <span
                        style={{
                          display:
                            "flex",
                          alignItems:
                            "center",
                          gap: "4px",
                          fontSize:
                            "11px",
                          color:
                            "#64748B",
                        }}
                      >
                        <GraduationCap
                          size={13}
                        />
                        {resource.level}
                      </span>
                    )}

                    {resource.duration && (
                      <span
                        style={{
                          display:
                            "flex",
                          alignItems:
                            "center",
                          gap: "4px",
                          fontSize:
                            "11px",
                          color:
                            "#64748B",
                        }}
                      >
                        <Clock
                          size={13}
                        />
                        {resource.duration}
                      </span>
                    )}

                    {resource.relevance !=
                      null && (
                      <span
                        style={{
                          fontSize:
                            "11px",
                          color:
                            "#3358E8",
                          fontWeight:
                            700,
                        }}
                      >
                        {Math.round(
                          resource.relevance
                        )}
                        % match
                      </span>
                    )}
                  </div>

                  {/* SKILLS */}

                  {resource.skills &&
                    resource.skills
                      .length >
                      0 && (
                      <div
                        style={{
                          display:
                            "flex",
                          gap: "5px",
                          flexWrap:
                            "wrap",
                          marginBottom:
                            "18px",
                        }}
                      >
                        {resource.skills
                          .slice(
                            0,
                            4
                          )
                          .map(
                            (
                              skill
                            ) => (
                              <span
                                key={
                                  skill
                                }
                                style={{
                                  fontSize:
                                    "10px",
                                  background:
                                    "#F8FAFC",
                                  border:
                                    "1px solid #E2E8F0",
                                  padding:
                                    "4px 7px",
                                  borderRadius:
                                    "5px",
                                }}
                              >
                                {skill}
                              </span>
                            )
                          )}
                      </div>
                    )}

                  {/* OPEN */}

                  <a
                    href={
                      resource.url
                    }
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      marginTop:
                        "auto",
                      display:
                        "flex",
                      alignItems:
                        "center",
                      justifyContent:
                        "center",
                      gap: "7px",
                      padding:
                        "11px",
                      borderRadius:
                        "9px",
                      background:
                        isYouTube
                          ? "#FEF2F2"
                          : "#EEF4FF",
                      color:
                        isYouTube
                          ? "#DC2626"
                          : "#3358E8",
                      textDecoration:
                        "none",
                      fontWeight:
                        700,
                      fontSize:
                        "13px",
                    }}
                  >
                    {isYouTube ? (
                      <PlayCircle
                        size={16}
                      />
                    ) : (
                      <ExternalLink
                        size={16}
                      />
                    )}

                    {isYouTube
                      ? "Watch on YouTube"
                      : "Open Course"}
                  </a>
                </div>
              );
            }
          )}
        </div>
      )}

      {/* RESOURCE COUNTS */}

      {!loading &&
        resources.length >
          0 && (
          <div
            style={{
              marginTop:
                "25px",
              display:
                "flex",
              gap: "18px",
              color:
                "#64748B",
              fontSize:
                "12px",
            }}
          >
            <span>
              <Globe
                size={13}
                style={{
                  verticalAlign:
                    "middle",
                  marginRight:
                    "4px",
                }}
              />
              {resources.length} live
              resources
            </span>

            <span>
              {youtubeResources.length}{" "}
              YouTube
            </span>

            <span>
              {courseResources.length}{" "}
              courses/resources
            </span>
          </div>
        )}
    </div>
  );
}